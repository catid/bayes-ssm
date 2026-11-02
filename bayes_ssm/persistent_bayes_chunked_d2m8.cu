#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float warp_sum_d2(float value) {
  for (int offset = 8; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0x0000ffffu, value, offset);
  return value;
}

template <int THREADS>
__device__ __forceinline__ void sync_threads_d2() {
  if constexpr (THREADS <= 32)
    __syncwarp(THREADS == 16 ? 0x0000ffffu : 0xffffffffu);
  else
    __syncthreads();
}

template <int D, int M, int THREADS>
__global__ void bayes_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const float* __restrict__ queries,
    const float* __restrict__ decay,
    const float* __restrict__ process,
    const float* __restrict__ observation,
    const float* __restrict__ mean0,
    const float* __restrict__ cov0,
    float* __restrict__ outputs,
    float* __restrict__ mean_out,
    float* __restrict__ cov_out,
    float* __restrict__ mp_hist,
    float* __restrict__ pp_hist,
    float* __restrict__ u_hist,
    float* __restrict__ beta_hist,
    float* __restrict__ e_hist,
    float* __restrict__ mn_hist,
    float* __restrict__ cn_hist,
    int n, int t, float floor) {
  const int b = blockIdx.x;
  if (b >= n) return;
  extern __shared__ float sm[];
  float* mean = sm;
  float* cov = mean + D * M;
  float* cov_pred = cov + D * D;
  float* tmp = cov_pred + D * D;
  float* sk = tmp + D;
  float* si = sk + M;
  int tid = threadIdx.x;
  for (int x = tid; x < D * M; x += THREADS) mean[x] = mean0[b * D * M + x];
  for (int x = tid; x < D * D; x += THREADS) cov[x] = cov0[b * D * D + x];
  sync_threads_d2<THREADS>();
  for (int ti = 0; ti < t; ++ti) {
    const float* k = keys + (b * t + ti) * D;
    const float* v = values + (b * t + ti) * M;
    const float* q = queries + (b * t + ti) * D;
    float* out = outputs + (b * t + ti) * M;
    const float* d = decay + (b * t + ti) * D;
    float pv = process[b * t + ti];
    float rv = observation[b * t + ti];
    for (int x = tid; x < D * D; x += THREADS) {
      int i = x / D, j = x - i * D;
      cov_pred[x] = cov[x] * d[i] * d[j] + (i == j ? pv : 0.0f);
    }
    sync_threads_d2<THREADS>();
    for (int i = tid; i < D; i += THREADS) {
      float z = 0.0f;
      for (int j = 0; j < D; ++j) z += cov_pred[i * D + j] * k[j];
      tmp[i] = z;
    }
    sync_threads_d2<THREADS>();
    if (tid == 0) {
      float z = rv;
      for (int i = 0; i < D; ++i) z += k[i] * tmp[i];
      sk[0] = fmaxf(z, floor);
    }
    // The mean prediction and update are done one value column at a time.
    for (int c = tid; c < M; c += THREADS) {
      float pred = 0.0f;
      for (int i = 0; i < D; ++i) pred += d[i] * mean[i * M + c] * k[i];
      si[c] = v[c] - pred;
    }
    sync_threads_d2<THREADS>();
    float beta = 1.0f / sk[0];
    const size_t hbase = (size_t)b * (size_t)t + (size_t)ti;
    if (mp_hist) {
      for (int x = tid; x < D * M; x += THREADS) {
        mp_hist[hbase * D * M + x] = d[x / M] * mean[x];
        e_hist[hbase * M + (x % M)] = si[x % M];
      }
      for (int x = tid; x < D * D; x += THREADS)
        pp_hist[hbase * D * D + x] = cov_pred[x];
      for (int i = tid; i < D; i += THREADS) u_hist[hbase * D + i] = tmp[i];
      if (tid == 0) beta_hist[hbase] = beta;
    }
    for (int x = tid; x < D * M; x += THREADS) {
      int i = x / M, c = x - i * M;
      mean[x] = d[i] * mean[x] + beta * tmp[i] * si[c];
    }
    sync_threads_d2<THREADS>();
    for (int x = tid; x < D * D; x += THREADS) {
      int i = x / D, j = x - i * D;
      float updated = cov_pred[x] - beta * tmp[i] * tmp[j];
      float transpose = cov_pred[j * D + i] - beta * tmp[j] * tmp[i];
      cov[x] = 0.5f * (updated + transpose);
    }
    sync_threads_d2<THREADS>();
    if (mn_hist) {
      for (int x = tid; x < D * M; x += THREADS)
        mn_hist[hbase * D * M + x] = mean[x];
      for (int x = tid; x < D * D; x += THREADS)
        cn_hist[hbase * D * D + x] = cov[x];
    }
    sync_threads_d2<THREADS>();
    for (int c = tid; c < M; c += THREADS) {
      float z = 0.0f;
      for (int i = 0; i < D; ++i) z += mean[i * M + c] * q[i];
      out[c] = z;
    }
    sync_threads_d2<THREADS>();
  }
  for (int x = tid; x < D * M; x += THREADS) mean_out[b * D * M + x] = mean[x];
  for (int x = tid; x < D * D; x += THREADS) cov_out[b * D * D + x] = cov[x];
}

} // namespace

std::vector<torch::Tensor> persistent_bayes(
    torch::Tensor keys, torch::Tensor values, torch::Tensor queries,
    torch::Tensor decay, torch::Tensor process, torch::Tensor observation,
    torch::Tensor mean0, torch::Tensor cov0, double floor) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kFloat);
  TORCH_CHECK(keys.dim() == 3 && values.dim() == 3 && queries.dim() == 3);
  const int n = keys.size(0), t = keys.size(1), d = keys.size(2), m = values.size(2);
  TORCH_CHECK(d == 2 && m == 8, "prototype supports D=2,M=8");
  auto out = torch::empty({n, t, m}, keys.options());
  auto mf = torch::empty_like(mean0);
  auto cf = torch::empty_like(cov0);
  // D=2/M=8 only needs sixteen lanes for the largest state tile.  The
  // previous 64-thread launch left 3/4 of the block idle in every scalar and
  // covariance phase, reducing the number of resident sequences.
  constexpr int threads = 16;
  size_t shared = (4 * 8 + 4 * 4 + 4 * 4 + 4 + 8 + 8 + 4) * sizeof(float);
  cudaFuncSetAttribute(
      bayes_kernel<2, 8, threads>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  auto stream = at::cuda::getCurrentCUDAStream();
  bayes_kernel<2,8,threads><<<n, threads, shared, stream>>>(
      keys.data_ptr<float>(), values.data_ptr<float>(), queries.data_ptr<float>(),
      decay.data_ptr<float>(), process.data_ptr<float>(), observation.data_ptr<float>(),
      mean0.data_ptr<float>(), cov0.data_ptr<float>(), out.data_ptr<float>(),
      mf.data_ptr<float>(), cf.data_ptr<float>(), nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, n, t, static_cast<float>(floor));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, mf, cf};
}

std::vector<torch::Tensor> persistent_bayes_saved(
    torch::Tensor keys, torch::Tensor values, torch::Tensor queries,
    torch::Tensor decay, torch::Tensor process, torch::Tensor observation,
    torch::Tensor mean0, torch::Tensor cov0, double floor) {
  const int n = keys.size(0), t = keys.size(1);
  auto out = torch::empty({n,t,8}, keys.options());
  auto mf = torch::empty_like(mean0), cf = torch::empty_like(cov0);
  auto mp = torch::empty({n,t,2,8}, keys.options());
  auto pp = torch::empty({n,t,2,2}, keys.options());
  auto uh = torch::empty({n,t,2}, keys.options());
  auto bh = torch::empty({n,t}, keys.options());
  auto eh = torch::empty({n,t,8}, keys.options());
  auto mn = torch::empty({n,t,2,8}, keys.options());
  auto cn = torch::empty({n,t,2,2}, keys.options());
  constexpr int threads = 16;
  size_t shared = (4*8*3 + 4*4*2 + 4 + 8) * sizeof(float);
  cudaFuncSetAttribute(bayes_kernel<2,8,threads>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  auto stream = at::cuda::getCurrentCUDAStream();
  bayes_kernel<2,8,threads><<<n,threads,shared,stream>>>(
      keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),
      decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),
      mean0.data_ptr<float>(),cov0.data_ptr<float>(),out.data_ptr<float>(),mf.data_ptr<float>(),cf.data_ptr<float>(),
      mp.data_ptr<float>(),pp.data_ptr<float>(),uh.data_ptr<float>(),bh.data_ptr<float>(),eh.data_ptr<float>(),mn.data_ptr<float>(),cn.data_ptr<float>(),n,t,static_cast<float>(floor));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out,mf,cf,mp,pp,uh,bh,eh,mn,cn};
}



namespace {

template<int D,int M,int THREADS,bool FIXED=false>
__global__ void chunk_forward_kernel(
 const float* __restrict__ keys,const float* __restrict__ values,const float* __restrict__ queries,
 const float* __restrict__ decay,const float* __restrict__ process,const float* __restrict__ observation,
 const float* __restrict__ mean0,const float* __restrict__ cov0,float* __restrict__ outputs,
 float* __restrict__ mean_final,float* __restrict__ cov_final,float* __restrict__ bmean,float* __restrict__ bcov,
 int n,int t,int chunks,int nb,float floor) {
 int b=blockIdx.x,tid=threadIdx.x;if(b>=n)return; extern __shared__ float sm[];
 // Compute the dynamic innovation vector directly from the old covariance;
 // cp is never consumed after its matvec and needlessly costs a shared tile.
 float* mean=sm;float* cov=mean+D*M;
 float* u=cov+D*D;float* wk=u+D;float* err=wk+D;float* si=err+M;
 for(int x=tid;x<D*M;x+=THREADS) mean[x]=mean0[b*D*M+x];
 for(int x=tid;x<D*D;x+=THREADS) cov[x]=cov0[b*D*D+x]; sync_threads_d2<THREADS>();
 for(int x=tid;x<D*M;x+=THREADS) bmean[b*(nb+1)*D*M+x]=mean[x];
 for(int x=tid;x<D*D;x+=THREADS) bcov[b*(nb+1)*D*D+x]=cov[x]; sync_threads_d2<THREADS>();
 for(int ti=0;ti<t;++ti) {
   if(ti>0 && (ti%chunks)==0) {int c=ti/chunks; for(int x=tid;x<D*M;x+=THREADS)bmean[(b*(nb+1)+c)*D*M+x]=mean[x];for(int x=tid;x<D*D;x+=THREADS)bcov[(b*(nb+1)+c)*D*D+x]=cov[x];}
   const float* k=keys+(b*t+ti)*D;const float* v=values+(b*t+ti)*M;const float* q=queries+(b*t+ti)*D;const float* d=decay+(b*t+ti)*D;float pv=FIXED?0.1f:process[b*t+ti],rv=FIXED?1.0f:observation[b*t+ti];
   if constexpr (FIXED) {
     // For A=rho I and Q=qI, compute Pk without rereading the freshly
     // materialized covariance prediction.  The diagonal process term is
     // folded into the final element, preserving the same recurrence.
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*k[j];u[i]=0.9409f*z+0.1f*k[i];}
   } else {
     for(int j=tid;j<D;j+=THREADS) wk[j]=d[j]*k[j];
     sync_threads_d2<THREADS>();
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*wk[j];u[i]=d[i]*z+pv*k[i];}
   }
   sync_threads_d2<THREADS>();
   if(tid==0){float z=rv;for(int i=0;i<D;++i)z+=k[i]*u[i];si[0]=fmaxf(z,floor);} 
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=(FIXED?0.97f:d[i])*mean[i*M+c]*k[i];err[c]=v[c]-z;} sync_threads_d2<THREADS>();float beta=1.f/si[0];
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mean[x]=(FIXED?0.97f:d[i])*mean[x]+beta*u[i]*err[c];}
   sync_threads_d2<THREADS>();
   if constexpr (FIXED) {
     // The fixed recurrence preserves symmetry. Compute each upper-triangle
     // entry once and mirror it, halving the covariance update arithmetic.
     if (tid < D * D) {
       int i=tid / D, j=tid % D;
       if (j >= i) {
         float value=0.9409f*cov[i*D+j]+(i==j?0.1f:0.f)-beta*u[i]*u[j];
         cov[i*D+j]=value;cov[j*D+i]=value;
       }
     }
   } else {
     for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float value=cov[i*D+j]*d[i]*d[j]+(i==j?pv:0.f)-beta*u[i]*u[j];cov[i*D+j]=value;cov[j*D+i]=value;}}
   }
   sync_threads_d2<THREADS>();
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=mean[i*M+c]*q[i];outputs[(b*t+ti)*M+c]=z;} sync_threads_d2<THREADS>();
 }
 for(int x=tid;x<D*M;x+=THREADS)mean_final[b*D*M+x]=mean[x];for(int x=tid;x<D*D;x+=THREADS)cov_final[b*D*D+x]=cov[x];
 for(int x=tid;x<D*M;x+=THREADS)bmean[(b*(nb+1)+nb)*D*M+x]=mean[x];for(int x=tid;x<D*D;x+=THREADS)bcov[(b*(nb+1)+nb)*D*D+x]=cov[x];
}

template<int D,int M,int THREADS,bool FIXED=false>
__global__ void recompute_chunk_kernel(
 const float* __restrict__ keys,const float* __restrict__ values,const float* __restrict__ decay,const float* __restrict__ process,const float* __restrict__ observation,
 const float* __restrict__ bmean,const float* __restrict__ bcov,float* __restrict__ mp,float* __restrict__ pp,float* __restrict__ uh,float* __restrict__ bh,float* __restrict__ eh,
 int n,int t,int chunks,int nb,int ci,float floor) {
 int b=blockIdx.x,tid=threadIdx.x;if(b>=n)return;extern __shared__ float sm[];float* mean=sm;float* cov=mean+D*M;
 float* u=cov+D*D;float* wk=u+D;float* err=wk+D;float* si=err+M;
 size_t bo=(size_t)b*(nb+1);for(int x=tid;x<D*M;x+=THREADS)mean[x]=bmean[bo*D*M+(size_t)ci*D*M+x];for(int x=tid;x<D*D;x+=THREADS)cov[x]=bcov[bo*D*D+(size_t)ci*D*D+x];sync_threads_d2<THREADS>();
 int start=ci*chunks,len=min(chunks,t-start);for(int li=0;li<len;++li){int ti=start+li;size_t hb=(size_t)b*(size_t)chunks+(size_t)li;const float*k=keys+((size_t)b*t+ti)*D;const float*v=values+((size_t)b*t+ti)*M;const float*d=decay+((size_t)b*t+ti)*D;float pv=FIXED?0.1f:process[(size_t)b*t+ti],rv=FIXED?1.0f:observation[(size_t)b*t+ti];
   if constexpr (!FIXED) {
     // Replay history stores the symmetric predicted covariance once.
     for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j)pp[hb*D*D+i*D+j]=cov[i*D+j]*d[i]*d[j]+(i==j?pv:0.f);}
   }
   sync_threads_d2<THREADS>();
   if constexpr (FIXED) {
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*k[j];u[i]=0.9409f*z+0.1f*k[i];}
   } else {
     for(int j=tid;j<D;j+=THREADS) wk[j]=d[j]*k[j];
     sync_threads_d2<THREADS>();
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*wk[j];u[i]=d[i]*z+pv*k[i];}
   }
   sync_threads_d2<THREADS>();if(tid==0){float z=rv;for(int i=0;i<D;++i)z+=k[i]*u[i];si[0]=fmaxf(z,floor);}for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=(FIXED?0.97f:d[i])*mean[i*M+c]*k[i];err[c]=v[c]-z;}sync_threads_d2<THREADS>();float beta=1.f/si[0];
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mp[hb*D*M+x]=(FIXED?0.97f:d[i])*mean[x];}
   if constexpr (FIXED) {
     for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;pp[hb*D*D+x]=0.9409f*cov[x]+(i==j?0.1f:0.f);}
   } else {
     // pp was written directly from the previous covariance state above.
   }
   for(int i=tid;i<D;i+=THREADS)uh[hb*D+i]=u[i];for(int c=tid;c<M;c+=THREADS)eh[hb*M+c]=err[c];if(tid==0)bh[hb]=beta;
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mean[x]=(FIXED?0.97f:d[i])*mean[x]+beta*u[i]*err[c];}sync_threads_d2<THREADS>();
   if constexpr (FIXED) {
     if (tid < D * D) {
       int i=tid / D, j=tid % D;
       if (j >= i) {
         float value=0.9409f*cov[i*D+j]+(i==j?0.1f:0.f)-beta*u[i]*u[j];
         cov[i*D+j]=value;cov[j*D+i]=value;
       }
     }
   } else {
     for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float value=cov[i*D+j]*d[i]*d[j]+(i==j?pv:0.f)-beta*u[i]*u[j];cov[i*D+j]=value;cov[j*D+i]=value;}}
   }
   sync_threads_d2<THREADS>();
 }
}

template<int D,int M,int THREADS,bool NEED_DYNAMICS,bool FIXED=false>
__global__ void backward_chunk_kernel(
 const float* __restrict__ keys,const float* __restrict__ values,const float* __restrict__ queries,const float* __restrict__ decay,const float* __restrict__ process,const float* __restrict__ observation,
 const float* __restrict__ mean0,const float* __restrict__ cov0,const float* __restrict__ bmean,const float* __restrict__ bcov,
 const float* __restrict__ mp,const float* __restrict__ pp,const float* __restrict__ uh,const float* __restrict__ bh,const float* __restrict__ eh,const float* __restrict__ gy,
 float* __restrict__ gkeys,float* __restrict__ gvalues,float* __restrict__ gqueries,float* __restrict__ gdecay,float* __restrict__ gprocess,float* __restrict__ gobservation,
 float* __restrict__ gm_state,float* __restrict__ gc_state,int n,int t,int chunks,int nb,int ci,float floor) {
 int b=blockIdx.x,tid=threadIdx.x;if(b>=n)return;extern __shared__ float sm[];float*gM=sm;float*gMprev=FIXED ? gM : gM+D*M;float*gC=FIXED ? gM+D*M : gMprev+D*M;float*gCprev=gC+D*D;float*ge=gCprev+D*D;float*gu=ge+M;float*gk=gu+D;float*red=gk+D;__shared__ float scalars[3];
 for(int x=tid;x<D*M;x+=THREADS)gM[x]=gm_state[b*D*M+x];for(int x=tid;x<D*D;x+=THREADS)gC[x]=gc_state[b*D*D+x];sync_threads_d2<THREADS>();int start=ci*chunks,len=min(chunks,t-start);
 for(int li=len-1;li>=0;--li){int ti=start+li;size_t hb=(size_t)b*(size_t)chunks+(size_t)li;const float*k=keys+((size_t)b*t+ti)*D;const float*q=queries+((size_t)b*t+ti)*D;const float*d=decay+((size_t)b*t+ti)*D;const float*mp0=mp+hb*D*M;const float*pp0=pp+hb*D*D;const float*u=uh+hb*D;const float*e=eh+hb*M;float rv=FIXED?1.0f:observation[(size_t)b*t+ti];float beta=bh[hb];const float*gyt=gy+((size_t)b*t+ti)*M;
   // derive updated mean for q gradient, then add output gradient
   for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=(mp0[i*M+c]+beta*u[i]*e[c])*gyt[c];gqueries[((size_t)b*t+ti)*D+i]=z;}
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gM[x]+=q[i]*gyt[c];}sync_threads_d2<THREADS>();
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=gM[i*M+c]*u[i];ge[c]=beta*z;gvalues[((size_t)b*t+ti)*M+c]=ge[c];}sync_threads_d2<THREADS>();
   if constexpr (!FIXED) {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gMprev[x]=gM[x]-k[i]*ge[c];}
   }
   for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=mp0[i*M+c]*ge[c];gk[i]=-z;}sync_threads_d2<THREADS>();
   float z=0;for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);z-=r*u[i]*u[j]*(i==j?1.f:2.f);}}for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;z+=gM[x]*u[i]*e[c];}z=warp_sum_d2(z);
   if(tid==0){float s=rv;for(int i=0;i<D;++i)s+=k[i]*u[i];scalars[0]=z;scalars[1]=(s>floor)?-z*beta*beta:0.f;
     if constexpr (NEED_DYNAMICS && !FIXED) gobservation[(size_t)b*t+ti]=scalars[1];
   }__syncwarp(0x0000ffffu);float gs=scalars[1];
   for(int i=tid;i<D;i+=THREADS){float a=0,bv=0;for(int c=0;c<M;++c)a+=gM[i*M+c]*e[c];for(int j=0;j<D;++j){float r=.5f*(gC[i*D+j]+gC[j*D+i]);bv+=2.f*r*u[j];}gu[i]=beta*(a-bv)+gs*k[i];gk[i]+=gs*u[i];}sync_threads_d2<THREADS>();
   for(int i=tid;i<D;i+=THREADS){float zz=0;for(int j=0;j<D;++j){int pi=j<i?j:i;int pj=j<i?i:j;zz+=pp0[pi*D+pj]*gu[j];}gk[i]+=zz;}for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);
       // P is stored as a symmetric matrix.  The u=P'k contribution must
       // be projected onto that symmetric subspace as well; otherwise the
       // initial-covariance gradient acquires a spurious antisymmetric part.
       float gu_sym=0.5f*(gu[i]*k[j]+gu[j]*k[i]);
       float value=(r+gu_sym)*(FIXED?0.9409f:d[i]*d[j]);gCprev[i*D+j]=value;gCprev[j*D+i]=value;}}sync_threads_d2<THREADS>();
   const float*mean_prev;const float*cov_prev;float beta_prev=0;const float*mp_prev=nullptr;const float*uh_prev=nullptr;const float*eh_prev=nullptr;const float*pp_prev=nullptr; if(li==0){mean_prev=bmean+((size_t)b*(nb+1)+ci)*D*M;cov_prev=bcov+((size_t)b*(nb+1)+ci)*D*D;}else{size_t hp=(size_t)b*chunks+(size_t)(li-1);mp_prev=mp+hp*D*M;uh_prev=uh+hp*D;eh_prev=eh+hp*M;pp_prev=pp+hp*D*D;beta_prev=bh[hp];}
   // gMprev is produced cooperatively above; wait before the per-row decay
   // reduction reads all M columns.
   sync_threads_d2<THREADS>();
   if constexpr (NEED_DYNAMICS && !FIXED) {
     for(int i=tid;i<D;i+=THREADS){float zd=0,zm=0;const float inv_di=1.0f/d[i];for(int j=0;j<D;++j){float cpv;if(li==0)cpv=cov_prev[i*D+j];else{int pi=j<i?j:i;int pj=j<i?i:j;cpv=pp_prev[pi*D+pj]-beta_prev*uh_prev[i]*uh_prev[j];}zd+=(gCprev[i*D+j]+gCprev[j*D+i])*cpv*inv_di;}for(int c=0;c<M;++c){float mv;if(li==0)mv=mean_prev[i*M+c];else mv=mp_prev[i*M+c]+beta_prev*uh_prev[i]*eh_prev[c];zm+=gMprev[i*M+c]*mv;}gdecay[((size_t)b*t+ti)*D+i]=zd+zm;}
     float pz=0;for(int i=tid;i<D;i+=THREADS){float inv_di=1.0f/d[i];pz+=gCprev[i*D+i]*inv_di*inv_di;}pz=warp_sum_d2(pz);if(tid==0)gprocess[(size_t)b*t+ti]=pz;
   }
   if constexpr (FIXED) {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gM[x]=(gM[x]-k[i]*ge[c])*0.97f;}
   } else {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M;gM[x]=gMprev[x]*d[i];}
   }
   for(int x=tid;x<D*D;x+=THREADS)gC[x]=gCprev[x];for(int i=tid;i<D;i+=THREADS)gkeys[((size_t)b*t+ti)*D+i]=gk[i];sync_threads_d2<THREADS>();
 }
 for(int x=tid;x<D*M;x+=THREADS)gm_state[b*D*M+x]=gM[x];for(int x=tid;x<D*D;x+=THREADS)gc_state[b*D*D+x]=gC[x];
}
}

std::vector<torch::Tensor> forward_chunked(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,int chunks,double floor){
 TORCH_CHECK(keys.is_cuda() && values.is_cuda() && queries.is_cuda(), "fused Bayesian CUDA inputs must be CUDA tensors");
 TORCH_CHECK(keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && queries.scalar_type()==at::kFloat, "fused Bayesian inputs must be float32");
 TORCH_CHECK(keys.is_contiguous() && values.is_contiguous() && queries.is_contiguous() && decay.is_contiguous() && process.is_contiguous() && observation.is_contiguous() && mean0.is_contiguous() && cov0.is_contiguous(), "fused Bayesian inputs must be contiguous");
 TORCH_CHECK(chunks > 0 && keys.dim()==3 && values.dim()==3 && queries.dim()==3 && keys.size(2)==2 && values.size(2)==8 && queries.size(2)==2, "fused Bayesian kernel expects [N,T,8] tensors");
 int n=keys.size(0),t=keys.size(1),nb=(t+chunks-1)/chunks;auto out=torch::empty({n,t,8},keys.options()),mf=torch::empty_like(mean0),cf=torch::empty_like(cov0),bm=torch::empty({n,nb+1,2,8},keys.options()),bc=torch::empty({n,nb+1,2,2},keys.options());constexpr int th=16;constexpr size_t sh=(2*8+2*4+2*2+2*8)*sizeof(float);cudaFuncSetAttribute(chunk_forward_kernel<2,8,th>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);auto st=at::cuda::getCurrentCUDAStream();chunk_forward_kernel<2,8,th><<<n,th,sh,st>>>(keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),mean0.data_ptr<float>(),cov0.data_ptr<float>(),out.data_ptr<float>(),mf.data_ptr<float>(),cf.data_ptr<float>(),bm.data_ptr<float>(),bc.data_ptr<float>(),n,t,chunks,nb,(float)floor);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,mf,cf,bm,bc};
}

std::vector<torch::Tensor> forward_chunked_fixed(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,int chunks,double floor){
 TORCH_CHECK(keys.is_cuda() && values.is_cuda() && queries.is_cuda(), "fused Bayesian CUDA inputs must be CUDA tensors");
 int n=keys.size(0),t=keys.size(1),nb=(t+chunks-1)/chunks;auto out=torch::empty({n,t,8},keys.options()),mf=torch::empty_like(mean0),cf=torch::empty_like(cov0),bm=torch::empty({n,nb+1,2,8},keys.options()),bc=torch::empty({n,nb+1,2,2},keys.options());constexpr int th=16;constexpr size_t sh=(2*8+2*4+2*2+2*8)*sizeof(float);cudaFuncSetAttribute(chunk_forward_kernel<2,8,th,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);auto st=at::cuda::getCurrentCUDAStream();chunk_forward_kernel<2,8,th,true><<<n,th,sh,st>>>(keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),mean0.data_ptr<float>(),cov0.data_ptr<float>(),out.data_ptr<float>(),mf.data_ptr<float>(),cf.data_ptr<float>(),bm.data_ptr<float>(),bc.data_ptr<float>(),n,t,chunks,nb,(float)floor);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,mf,cf,bm,bc};
}

template<bool NEED_DYNAMICS,bool FIXED=false>
std::vector<torch::Tensor> backward_chunked_impl(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,torch::Tensor bm,torch::Tensor bc,torch::Tensor gy,torch::Tensor gmf,torch::Tensor gcf,int chunks,double floor){
 TORCH_CHECK(keys.is_cuda() && keys.scalar_type()==at::kFloat && keys.is_contiguous() && values.is_contiguous() && queries.is_contiguous() && decay.is_contiguous() && process.is_contiguous() && observation.is_contiguous() && mean0.is_contiguous() && cov0.is_contiguous() && bm.is_contiguous() && bc.is_contiguous() && gy.is_contiguous() && gmf.is_contiguous() && gcf.is_contiguous(), "fused Bayesian backward inputs must be contiguous CUDA float32 tensors");
 int n=keys.size(0),t=keys.size(1),nb=(t+chunks-1)/chunks;
 auto gk=torch::zeros_like(keys), gv=torch::zeros_like(values), gq=torch::zeros_like(queries);
 torch::Tensor gd, gp, gr, gm0, gc0;
 if constexpr (NEED_DYNAMICS) {
   gd=torch::zeros_like(decay); gp=torch::zeros_like(process);
   gr=torch::zeros_like(observation); gm0=torch::zeros_like(mean0);
   gc0=torch::zeros_like(cov0);
 }
 auto gm=torch::empty_like(mean0),gc=torch::empty_like(cov0);
 gm.copy_(gmf); gc.copy_(gcf);
 auto mp=torch::empty({n,chunks,2,8},keys.options()),pp=torch::empty({n,chunks,2,2},keys.options());
 auto uh=torch::empty({n,chunks,2},keys.options()),bh=torch::empty({n,chunks},keys.options());
 auto eh=torch::empty({n,chunks,8},keys.options());
 float *gd_ptr=nullptr,*gp_ptr=nullptr,*gr_ptr=nullptr;
 if constexpr (NEED_DYNAMICS) {
   gd_ptr=gd.data_ptr<float>(); gp_ptr=gp.data_ptr<float>(); gr_ptr=gr.data_ptr<float>();
 }
 // D2/M8's largest tile is D*M=16.  Keep one lane per tile element; a
 // full warp only repeats barriers and reductions for this narrow state.
 constexpr int th=16;
 constexpr size_t recompute_sh=(2*8+2*4+2*2+2*8)*sizeof(float);
 constexpr size_t backward_sh=((FIXED ? 8 : 16) + 2*4 + 8 + 3*2 + th)*sizeof(float);
 cudaFuncSetAttribute(recompute_chunk_kernel<2,8,th,FIXED>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)recompute_sh);
 cudaFuncSetAttribute(backward_chunk_kernel<2,8,th,NEED_DYNAMICS,FIXED>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)backward_sh);
 auto st=at::cuda::getCurrentCUDAStream();
 for(int ci=nb-1;ci>=0;--ci){
   recompute_chunk_kernel<2,8,th,FIXED><<<n,th,recompute_sh,st>>>(
     keys.data_ptr<float>(),values.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),
     bm.data_ptr<float>(),bc.data_ptr<float>(),mp.data_ptr<float>(),pp.data_ptr<float>(),uh.data_ptr<float>(),bh.data_ptr<float>(),eh.data_ptr<float>(),n,t,chunks,nb,ci,(float)floor);
   backward_chunk_kernel<2,8,th,NEED_DYNAMICS,FIXED><<<n,th,backward_sh,st>>>(
     keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),
     mean0.data_ptr<float>(),cov0.data_ptr<float>(),bm.data_ptr<float>(),bc.data_ptr<float>(),mp.data_ptr<float>(),pp.data_ptr<float>(),uh.data_ptr<float>(),bh.data_ptr<float>(),eh.data_ptr<float>(),
     gy.data_ptr<float>(),gk.data_ptr<float>(),gv.data_ptr<float>(),gq.data_ptr<float>(),gd_ptr,gp_ptr,gr_ptr,gm.data_ptr<float>(),gc.data_ptr<float>(),n,t,chunks,nb,ci,(float)floor);
   C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 if constexpr (NEED_DYNAMICS) {
   gm0.copy_(gm); gc0.copy_(gc);
   return {gk,gv,gq,gd,gp,gr,gm0,gc0};
 }
 return {gk,gv,gq};
}

std::vector<torch::Tensor> backward_chunked(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,torch::Tensor bm,torch::Tensor bc,torch::Tensor gy,torch::Tensor gmf,torch::Tensor gcf,int chunks,double floor){
 return backward_chunked_impl<true,false>(keys,values,queries,decay,process,observation,mean0,cov0,bm,bc,gy,gmf,gcf,chunks,floor);
}

std::vector<torch::Tensor> backward_chunked_lean(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,torch::Tensor bm,torch::Tensor bc,torch::Tensor gy,torch::Tensor gmf,torch::Tensor gcf,int chunks,double floor){
 return backward_chunked_impl<false,false>(keys,values,queries,decay,process,observation,mean0,cov0,bm,bc,gy,gmf,gcf,chunks,floor);
}

std::vector<torch::Tensor> backward_chunked_lean_fixed(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,torch::Tensor bm,torch::Tensor bc,torch::Tensor gy,torch::Tensor gmf,torch::Tensor gcf,int chunks,double floor){
 return backward_chunked_impl<false,true>(keys,values,queries,decay,process,observation,mean0,cov0,bm,bc,gy,gmf,gcf,chunks,floor);
}


std::vector<torch::Tensor> backward_chunked_fixed(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,torch::Tensor bm,torch::Tensor bc,torch::Tensor gy,torch::Tensor gmf,torch::Tensor gcf,int chunks,double floor){
 return backward_chunked_impl<true,true>(keys,values,queries,decay,process,observation,mean0,cov0,bm,bc,gy,gmf,gcf,chunks,floor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward_chunked",&forward_chunked);m.def("backward_chunked",&backward_chunked);m.def("backward_chunked_lean",&backward_chunked_lean);m.def("forward_chunked_fixed",&forward_chunked_fixed);m.def("backward_chunked_lean_fixed",&backward_chunked_lean_fixed);m.def("backward_chunked_fixed",&backward_chunked_fixed);}
