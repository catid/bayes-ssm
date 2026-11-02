#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace {

__device__ __forceinline__ float warp_sum_d8(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// The fixed D8 reverse kernel uses 64 lanes.  Reduce each warp with shuffle,
// then combine the two lane-0 partials in the first warp.  This replaces the
// six full-block shared-memory reduction barriers in the old 256-thread path
// while preserving deterministic lane-0 ownership of the scalar results.
template <int THREADS>
__device__ __forceinline__ float block_sum_d8(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum_d8(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = 0.0f;
  if (warp == 0) {
    total = (lane < (THREADS / 32)) ? scratch[lane] : 0.0f;
    total = warp_sum_d8(total);
  }
  return total;
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
  __syncthreads();
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
    __syncthreads();
    for (int i = tid; i < D; i += THREADS) {
      float z = 0.0f;
      for (int j = 0; j < D; ++j) z += cov_pred[i * D + j] * k[j];
      tmp[i] = z;
    }
    __syncthreads();
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
    __syncthreads();
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
    __syncthreads();
    for (int x = tid; x < D * D; x += THREADS) {
      int i = x / D, j = x - i * D;
      float updated = cov_pred[x] - beta * tmp[i] * tmp[j];
      float transpose = cov_pred[j * D + i] - beta * tmp[j] * tmp[i];
      cov[x] = 0.5f * (updated + transpose);
    }
    __syncthreads();
    if (mn_hist) {
      for (int x = tid; x < D * M; x += THREADS)
        mn_hist[hbase * D * M + x] = mean[x];
      for (int x = tid; x < D * D; x += THREADS)
        cn_hist[hbase * D * D + x] = cov[x];
    }
    __syncthreads();
    for (int c = tid; c < M; c += THREADS) {
      float z = 0.0f;
      for (int i = 0; i < D; ++i) z += mean[i * M + c] * q[i];
      out[c] = z;
    }
    __syncthreads();
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
  TORCH_CHECK(d == 8 && m == 8, "prototype supports D=M=8");
  auto out = torch::empty({n, t, m}, keys.options());
  auto mf = torch::empty_like(mean0);
  auto cf = torch::empty_like(cov0);
  constexpr int threads = 64;
  size_t shared = (8 * 8 + 8 * 8 + 8 * 8 + 8 + 8 + 8 + 8) * sizeof(float);
  cudaFuncSetAttribute(
      bayes_kernel<8, 8, threads>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(shared));
  auto stream = at::cuda::getCurrentCUDAStream();
  bayes_kernel<8,8,threads><<<n, threads, shared, stream>>>(
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
  auto mp = torch::empty({n,t,8,8}, keys.options());
  auto pp = torch::empty({n,t,8,8}, keys.options());
  auto uh = torch::empty({n,t,8}, keys.options());
  auto bh = torch::empty({n,t}, keys.options());
  auto eh = torch::empty({n,t,8}, keys.options());
  auto mn = torch::empty({n,t,8,8}, keys.options());
  auto cn = torch::empty({n,t,8,8}, keys.options());
  constexpr int threads = 64;
  size_t shared = (8*8*3 + 8*4) * sizeof(float);
  cudaFuncSetAttribute(bayes_kernel<8,8,threads>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  auto stream = at::cuda::getCurrentCUDAStream();
  bayes_kernel<8,8,threads><<<n,threads,shared,stream>>>(
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
 // Keep only the recurrent state and the innovation workspace in shared
 // memory.  The old dynamic path materialized ``cp = A P A' + Q`` and then
 // immediately multiplied it by k to obtain u.  No forward consumer needs
 // cp after that product (the reverse path recomputes it), so compute u
 // directly from the old covariance and update cov in place below.  This
 // removes 64 shared stores/loads per token for D=8 and lowers the shared
 // footprint enough to improve occupancy on Blackwell.
 float* mean=sm;float* cov=mean+D*M;
 float* u=cov+D*D;float* wk=u+D;float* err=wk+D;float* si=err+M;
 for(int x=tid;x<D*M;x+=THREADS) mean[x]=mean0[b*D*M+x];
 for(int x=tid;x<D*D;x+=THREADS) cov[x]=cov0[b*D*D+x]; __syncthreads();
 for(int x=tid;x<D*M;x+=THREADS) bmean[b*(nb+1)*D*M+x]=mean[x];
 for(int x=tid;x<D*D;x+=THREADS) bcov[b*(nb+1)*D*D+x]=cov[x]; __syncthreads();
 for(int ti=0;ti<t;++ti) {
   if(ti>0 && (ti%chunks)==0) {int c=ti/chunks; for(int x=tid;x<D*M;x+=THREADS)bmean[(b*(nb+1)+c)*D*M+x]=mean[x];for(int x=tid;x<D*D;x+=THREADS)bcov[(b*(nb+1)+c)*D*D+x]=cov[x];}
   const float* k=keys+(b*t+ti)*D;const float* v=values+(b*t+ti)*M;const float* q=queries+(b*t+ti)*D;const float* d=decay+(b*t+ti)*D;float pv=FIXED?0.1f:process[b*t+ti],rv=FIXED?1.0f:observation[b*t+ti];
   if constexpr (FIXED) {
     // For A=rho I and Q=qI, compute Pk without rereading the freshly
     // materialized covariance prediction.  The diagonal process term is
     // folded into the final element, preserving the same recurrence.
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*k[j];u[i]=0.9409f*z+0.1f*k[i];}
   } else {
     // u = (A P A' + Q) k, without materializing the temporary covariance.
     // Factoring d[i] outside the inner loop keeps the operation count equal
     // to the cp+matvec sequence while avoiding a shared-memory round trip.
     // Reuse d[j] * k[j] across all covariance rows.  The direct fused
     // matvec otherwise recomputes those eight products once per row.
     for(int j=tid;j<D;j+=THREADS) wk[j]=d[j]*k[j];
     __syncthreads();
     for(int i=tid;i<D;i+=THREADS){
       float z=0.0f;
       for(int j=0;j<D;++j) z += cov[i*D+j] * wk[j];
       u[i] = d[i] * z + pv * k[i];
     }
   }
   __syncthreads();
   if(tid==0){float z=rv;for(int i=0;i<D;++i)z+=k[i]*u[i];si[0]=fmaxf(z,floor);} 
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i){if constexpr (FIXED) z+=0.97f*mean[i*M+c]*k[i]; else z+=mean[i*M+c]*wk[i];}err[c]=v[c]-z;} __syncthreads();float beta=1.f/si[0];
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mean[x]=(FIXED?0.97f:d[i])*mean[x]+beta*u[i]*err[c];}
   __syncthreads();
   if constexpr (FIXED) {
     // The fixed recurrence preserves symmetry. Compute each upper-triangle
     // entry once and mirror it, halving the covariance update arithmetic.
     // Assign each upper-triangle entry to exactly one thread.  The previous
     // row/lane mapping duplicated off-diagonal writers and raced.
     for (int flat = tid; flat < D * D; flat += THREADS) {
       int i = flat / D, j = flat - i * D;
       if (i <= j) {
         float value = 0.9409f * cov[i * D + j] + (i == j ? 0.1f : 0.0f)
                     - beta * u[i] * u[j];
         cov[i * D + j] = value;
         cov[j * D + i] = value;
       }
     }
   } else {
     // cp was intentionally not materialized above; update the old covariance
     // in place now that u has been computed from it.
     for(int x=tid;x<D*D;x+=THREADS){
       int i=x/D,j=x%D;
       if(i<=j){
         float value=cov[i*D+j]*d[i]*d[j] + (i==j?pv:0.f)
                    - beta*u[i]*u[j];
         cov[i*D+j]=value; cov[j*D+i]=value;
       }
     }
   }
   __syncthreads();
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=mean[i*M+c]*q[i];outputs[(b*t+ti)*M+c]=z;} __syncthreads();
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
 // The replay kernel needs P_pred only as a saved global boundary history.
 // Write that history directly and avoid allocating a shared ``cp`` tile;
 // the reverse pass consumes pp, while the forward update can reconstruct
 // each covariance entry from the old cov and diagonal dynamics.
 float* u=cov+D*D;float* wk=u+D;float* err=wk+D;float* si=err+M;
 size_t bo=(size_t)b*(nb+1);for(int x=tid;x<D*M;x+=THREADS)mean[x]=bmean[bo*D*M+(size_t)ci*D*M+x];for(int x=tid;x<D*D;x+=THREADS)cov[x]=bcov[bo*D*D+(size_t)ci*D*D+x];__syncthreads();
 int start=ci*chunks,len=min(chunks,t-start);for(int li=0;li<len;++li){int ti=start+li;size_t hb=(size_t)b*(size_t)chunks+(size_t)li;const float*k=keys+((size_t)b*t+ti)*D;const float*v=values+((size_t)b*t+ti)*M;const float*d=decay+((size_t)b*t+ti)*D;float pv=FIXED?0.1f:process[(size_t)b*t+ti],rv=FIXED?1.0f:observation[(size_t)b*t+ti];
   if constexpr (!FIXED) {
     for(int x=tid;x<D*D;x+=THREADS){
       int i=x/D,j=x%D;
       if (i <= j) pp[hb*D*D+i*D+j] = cov[i*D+j]*d[i]*d[j] + (i==j?pv:0.f);
     }
   }
   __syncthreads();
   if constexpr (FIXED) {
     for(int i=tid;i<D;i+=THREADS){float z=0;for(int j=0;j<D;++j)z+=cov[i*D+j]*k[j];u[i]=0.9409f*z+0.1f*k[i];}
   } else {
     for(int j=tid;j<D;j+=THREADS) wk[j]=d[j]*k[j];
     __syncthreads();
     for(int i=tid;i<D;i+=THREADS){
       float z=0.0f;
       for(int j=0;j<D;++j) z += cov[i*D+j]*wk[j];
       u[i]=d[i]*z + pv*k[i];
     }
   }
   __syncthreads();if(tid==0){float z=rv;for(int i=0;i<D;++i)z+=k[i]*u[i];si[0]=fmaxf(z,floor);}for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i){if constexpr (FIXED) z+=0.97f*mean[i*M+c]*k[i]; else z+=mean[i*M+c]*wk[i];}err[c]=v[c]-z;}__syncthreads();float beta=1.f/si[0];
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mp[hb*D*M+x]=(FIXED?0.97f:d[i])*mean[x];}
   if constexpr (FIXED) {
     for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;pp[hb*D*D+x]=0.9409f*cov[x]+(i==j?0.1f:0.f);}
   } else {
     // pp was written before the u reduction above, while cov still held
     // the previous state.  No local covariance prediction is needed.
     // (The branch is intentionally empty; retaining it documents that the
     // history has already been materialized.)
   }
   for(int i=tid;i<D;i+=THREADS)uh[hb*D+i]=u[i];for(int c=tid;c<M;c+=THREADS)eh[hb*M+c]=err[c];if(tid==0)bh[hb]=beta;
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mean[x]=(FIXED?0.97f:d[i])*mean[x]+beta*u[i]*err[c];}__syncthreads();
   if constexpr (FIXED) {
     for (int flat = tid; flat < D * D; flat += THREADS) {
       int i = flat / D, j = flat - i * D;
       if (i <= j) {
         float value = 0.9409f * cov[i * D + j] + (i == j ? 0.1f : 0.0f)
                     - beta * u[i] * u[j];
         cov[i * D + j] = value;
         cov[j * D + i] = value;
       }
     }
   } else {
     for(int x=tid;x<D*D;x+=THREADS){
       int i=x/D,j=x%D;
       if(i<=j){
         float value=cov[i*D+j]*d[i]*d[j] + (i==j?pv:0.f)
                    - beta*u[i]*u[j];
         cov[i*D+j]=value; cov[j*D+i]=value;
       }
     }
   }
   __syncthreads();
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
 for(int x=tid;x<D*M;x+=THREADS)gM[x]=gm_state[b*D*M+x];for(int x=tid;x<D*D;x+=THREADS)gC[x]=gc_state[b*D*D+x];__syncthreads();int start=ci*chunks,len=min(chunks,t-start);
 for(int li=len-1;li>=0;--li){int ti=start+li;size_t hb=(size_t)b*(size_t)chunks+(size_t)li;const float*k=keys+((size_t)b*t+ti)*D;const float*q=queries+((size_t)b*t+ti)*D;const float*d=decay+((size_t)b*t+ti)*D;const float*mp0=mp+hb*D*M;const float*pp0=pp+hb*D*D;const float*u=uh+hb*D;const float*e=eh+hb*M;float rv=FIXED?1.0f:observation[(size_t)b*t+ti];float beta=bh[hb];const float*gyt=gy+((size_t)b*t+ti)*M;
   // derive updated mean for q gradient, then add output gradient
   for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=(mp0[i*M+c]+beta*u[i]*e[c])*gyt[c];gqueries[((size_t)b*t+ti)*D+i]=z;}
   for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gM[x]+=q[i]*gyt[c];}__syncthreads();
   for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=gM[i*M+c]*u[i];ge[c]=beta*z;gvalues[((size_t)b*t+ti)*M+c]=ge[c];}__syncthreads();
   if constexpr (!FIXED) {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gMprev[x]=gM[x]-k[i]*ge[c];}
   }
   for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=mp0[i*M+c]*ge[c];gk[i]=-z;}__syncthreads();
   float z=0;for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);z-=r*u[i]*u[j]*(i==j?1.0f:2.0f);}}for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;z+=gM[x]*u[i]*e[c];}
   if constexpr (!FIXED) {
     z=warp_sum_d8(z);
     if(tid==0){float s=rv;for(int i=0;i<D;++i)s+=k[i]*u[i];scalars[0]=z;scalars[1]=(s>floor)?-z*beta*beta:0.f;
       if constexpr (NEED_DYNAMICS) gobservation[(size_t)b*t+ti]=scalars[1];
     }
     __syncwarp();
   } else {
     const float total = block_sum_d8<THREADS>(z, red);
     if(tid==0){float s=rv;for(int i=0;i<D;++i)s+=k[i]*u[i];scalars[0]=total;scalars[1]=(s>floor)?-total*beta*beta:0.f;}
     __syncthreads();
   }
   float gs=scalars[1];
   for(int i=tid;i<D;i+=THREADS){float a=0,bv=0;for(int c=0;c<M;++c)a+=gM[i*M+c]*e[c];for(int j=0;j<D;++j){float r=.5f*(gC[i*D+j]+gC[j*D+i]);bv+=2.f*r*u[j];}gu[i]=beta*(a-bv)+gs*k[i];gk[i]+=gs*u[i];}__syncthreads();
   for(int i=tid;i<D;i+=THREADS){float zz=0;for(int j=0;j<D;++j){int pi = j < i ? j : i; int pj = j < i ? i : j; zz+=pp0[pi*D+pj]*gu[j];}gk[i]+=zz;}for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);
       // P is stored as a symmetric matrix.  The u=P'k contribution must
       // be projected onto that symmetric subspace as well; otherwise the
       // initial-covariance gradient acquires a spurious antisymmetric part.
       float gu_sym=0.5f*(gu[i]*k[j]+gu[j]*k[i]);
       float value=(r+gu_sym)*(FIXED?0.9409f:d[i]*d[j]);
       gCprev[i*D+j]=value; gCprev[j*D+i]=value;}}__syncthreads();
   const float*mean_prev;const float*cov_prev;float beta_prev=0;const float*mp_prev=nullptr;const float*uh_prev=nullptr;const float*eh_prev=nullptr;const float*pp_prev=nullptr; if(li==0){mean_prev=bmean+((size_t)b*(nb+1)+ci)*D*M;cov_prev=bcov+((size_t)b*(nb+1)+ci)*D*D;}else{size_t hp=(size_t)b*chunks+(size_t)(li-1);mp_prev=mp+hp*D*M;uh_prev=uh+hp*D;eh_prev=eh+hp*M;pp_prev=pp+hp*D*D;beta_prev=bh[hp];}
   // gMprev is produced cooperatively above; wait before the per-row decay
   // reduction reads all M columns.
   __syncthreads();
   if constexpr (NEED_DYNAMICS && !FIXED) {
     // For diagonal A, dP_pred/dd_i is row/column P_prev*d_j.  Since
     // gCprev already contains the matching d_i*d_j factor, the two terms
     // reduce to (gC_ij + gC_ji) * P_pred_ij / d_i.  Hoisting one reciprocal
     // per row removes two divisions per covariance entry from the reverse
     // pass without changing the dense-covariance gradient.
     for(int i=tid;i<D;i+=THREADS){float zd=0,zm=0;const float inv_di=1.0f/d[i];for(int j=0;j<D;++j){int pi=j<i?j:i;int pj=j<i?i:j;float cpv;if(li==0)cpv=cov_prev[i*D+j];else cpv=pp_prev[pi*D+pj]-beta_prev*uh_prev[i]*uh_prev[j];zd+=(gCprev[i*D+j]+gCprev[j*D+i])*cpv*inv_di;}for(int c=0;c<M;++c){float mv;if(li==0)mv=mean_prev[i*M+c];else mv=mp_prev[i*M+c]+beta_prev*uh_prev[i]*eh_prev[c];zm+=gMprev[i*M+c]*mv;}gdecay[((size_t)b*t+ti)*D+i]=zd+zm;}
     float pz=0;for(int i=tid;i<D;i+=THREADS){float inv_di=1.0f/d[i];pz+=gCprev[i*D+i]*inv_di*inv_di;}if constexpr (!FIXED){pz=warp_sum_d8(pz);if(tid==0)gprocess[(size_t)b*t+ti]=pz;}else{const float total_p=block_sum_d8<THREADS>(pz, red);if(tid==0)gprocess[(size_t)b*t+ti]=total_p;}
   }
   if constexpr (FIXED) {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gM[x]=(gM[x]-k[i]*ge[c])*0.97f;}
   } else {
     for(int x=tid;x<D*M;x+=THREADS){int i=x/M;gM[x]=gMprev[x]*d[i];}
   }
   for(int x=tid;x<D*D;x+=THREADS)gC[x]=gCprev[x];for(int i=tid;i<D;i+=THREADS)gkeys[((size_t)b*t+ti)*D+i]=gk[i];__syncthreads();
 }
 for(int x=tid;x<D*M;x+=THREADS)gm_state[b*D*M+x]=gM[x];for(int x=tid;x<D*D;x+=THREADS)gc_state[b*D*D+x]=gC[x];
}

// Fused recompute + reverse kernel.  The regular path launches a recompute
// kernel and then a backward kernel for every chunk, materializing five
// history tensors in global memory between the launches.  For D=M=8 those
// histories fit comfortably in shared memory at the production chunk size
// (64), so keep them on chip and run the reverse pass immediately.  This
// removes the global history traffic and one launch per chunk while retaining
// the exact dense-covariance recurrence and all dynamic gradients.
template<int D,int M,int THREADS,bool NEED_DYNAMICS>
__global__ void fused_backward_chunk_kernel(
 const float* __restrict__ keys,const float* __restrict__ values,const float* __restrict__ queries,
 const float* __restrict__ decay,const float* __restrict__ process,const float* __restrict__ observation,
 const float* __restrict__ bmean,const float* __restrict__ bcov,
 const float* __restrict__ gy,
 float* __restrict__ gkeys,float* __restrict__ gvalues,float* __restrict__ gqueries,
 float* __restrict__ gdecay,float* __restrict__ gprocess,float* __restrict__ gobservation,
 float* __restrict__ gm_state,float* __restrict__ gc_state,
 int n,int t,int chunks,int nb,int ci,float floor) {
  const int b=blockIdx.x, tid=threadIdx.x;
  if (b>=n) return;
  extern __shared__ float sm[];
  // State and temporary vectors/matrices.  The predicted covariance is kept
  // directly in the per-token shared history (hpp); there is no need for a
  // second ``cp`` tile because the reverse pass consumes hpp and the forward
  // update can read it after the innovation vector has been formed.
  float* mean=sm; float* cov=mean+D*M;
  float* u=cov+D*D; float* err=u+D; float* gM=err+M;
  float* gC=gM+D*M; float* gMprev=gC+D*D; float* gCprev=gMprev+D*M;
  float* ge=gCprev+D*D; float* gu=ge+M; float* gk=gu+D;
  float* red=gk+D;
  // Histories are laid out with a runtime stride of `chunks`; only the first
  // len entries are used by the final (short) chunk.
  float* hmp=red+THREADS;
  float* hpp=hmp+(size_t)chunks*D*M;
  float* huh=hpp+(size_t)chunks*D*D;
  float* hbh=huh+(size_t)chunks*D;
  float* heh=hbh+chunks;
  __shared__ float scalars[3];
  const int start=ci*chunks, len=min(chunks,t-start);
  const size_t bo=((size_t)b*(nb+1)+ci);
  for(int x=tid;x<D*M;x+=THREADS){mean[x]=bmean[bo*D*M+x];gM[x]=gm_state[(size_t)b*D*M+x];}
  for(int x=tid;x<D*D;x+=THREADS){cov[x]=bcov[bo*D*D+x];gC[x]=gc_state[(size_t)b*D*D+x];}
  __syncthreads();
  // Forward replay into shared histories.
  for(int li=0;li<len;++li){
    const int ti=start+li;
    const float* k=keys+((size_t)b*t+ti)*D;
    const float* v=values+((size_t)b*t+ti)*M;
    const float* d=decay+((size_t)b*t+ti)*D;
    float pv=process[(size_t)b*t+ti], rv=observation[(size_t)b*t+ti];
    float* hpp_t = hpp+(size_t)li*D*D;
    for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;hpp_t[x]=cov[i*D+j]*d[i]*d[j]+(i==j?pv:0.f);}
    __syncthreads();
    for(int i=tid;i<D;i+=THREADS){float z=0.f;for(int j=0;j<D;++j)z+=cov[i*D+j]*d[j]*k[j];u[i]=d[i]*z+pv*k[i];}
    __syncthreads();
    if(tid==0){float z=rv;for(int i=0;i<D;++i)z+=k[i]*u[i];scalars[0]=fmaxf(z,floor);hbh[li]=1.f/scalars[0];}
    for(int c=tid;c<M;c+=THREADS){float z=0.f;for(int i=0;i<D;++i)z+=d[i]*mean[i*M+c]*k[i];err[c]=v[c]-z;}
    __syncthreads();
    const float beta=hbh[li];
    for(int x=tid;x<D*M;x+=THREADS){int i=x/M;hmp[(size_t)li*D*M+x]=d[i]*mean[x];heh[(size_t)li*M+(x%M)]=err[x%M];}
    for(int i=tid;i<D;i+=THREADS)huh[(size_t)li*D+i]=u[i];
    for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;mean[x]=d[i]*mean[x]+beta*u[i]*err[c];}
    __syncthreads();
    for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float z=hpp_t[x]-beta*u[i]*u[j];cov[i*D+j]=z;cov[j*D+i]=z;}}
    __syncthreads();
  }
  // Reverse replay.  This is algebraically identical to backward_chunk_kernel
  // but reads the histories above instead of global temporary tensors.
  for(int li=len-1;li>=0;--li){
    const int ti=start+li;
    const float* k=keys+((size_t)b*t+ti)*D;
    const float* q=queries+((size_t)b*t+ti)*D;
    const float* d=decay+((size_t)b*t+ti)*D;
    const float* mp0=hmp+(size_t)li*D*M; const float* pp0=hpp+(size_t)li*D*D;
    const float* uh=huh+(size_t)li*D; const float* e=heh+(size_t)li*M; const float beta=hbh[li];
    const float* gyt=gy+((size_t)b*t+ti)*M; const float rv=observation[(size_t)b*t+ti];
    for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=(mp0[i*M+c]+beta*uh[i]*e[c])*gyt[c];gqueries[((size_t)b*t+ti)*D+i]=z;}
    for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gM[x]+=q[i]*gyt[c];}
    __syncthreads();
    for(int c=tid;c<M;c+=THREADS){float z=0;for(int i=0;i<D;++i)z+=gM[i*M+c]*uh[i];ge[c]=beta*z;gvalues[((size_t)b*t+ti)*M+c]=ge[c];}
    __syncthreads();
    for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;gMprev[x]=gM[x]-k[i]*ge[c];}
    for(int i=tid;i<D;i+=THREADS){float z=0;for(int c=0;c<M;++c)z+=mp0[i*M+c]*ge[c];gk[i]=-z;}
    __syncthreads();
    float z=0;for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);z-=r*uh[i]*uh[j]*(i==j?1.0f:2.0f);}}for(int x=tid;x<D*M;x+=THREADS){int i=x/M,c=x%M;z+=gM[x]*uh[i]*e[c];}red[tid]=z;__syncthreads();
    for(int off=THREADS/2;off>0;off>>=1){if(tid<off)red[tid]+=red[tid+off];__syncthreads();}
    if(tid==0){float s=rv;for(int i=0;i<D;++i)s+=k[i]*uh[i];scalars[0]=red[0];scalars[1]=(s>floor)?-red[0]*beta*beta:0.f;if constexpr(NEED_DYNAMICS)gobservation[(size_t)b*t+ti]=scalars[1];}
    __syncthreads(); const float gs=scalars[1];
    for(int i=tid;i<D;i+=THREADS){float a=0,bv=0;for(int c=0;c<M;++c)a+=gM[i*M+c]*e[c];for(int j=0;j<D;++j){float r=.5f*(gC[i*D+j]+gC[j*D+i]);bv+=2.f*r*uh[j];}gu[i]=beta*(a-bv)+gs*k[i];gk[i]+=gs*uh[i];}
    __syncthreads();
    for(int i=tid;i<D;i+=THREADS){float zz=0;for(int j=0;j<D;++j){int pi=j<i?j:i;int pj=j<i?i:j;zz+=pp0[pi*D+pj]*gu[j];}gk[i]+=zz;}
    for(int x=tid;x<D*D;x+=THREADS){int i=x/D,j=x%D;if(i<=j){float r=.5f*(gC[x]+gC[j*D+i]);float gu_sym=.5f*(gu[i]*k[j]+gu[j]*k[i]);float value=(r+gu_sym)*d[i]*d[j];gCprev[i*D+j]=value;gCprev[j*D+i]=value;}}
    __syncthreads();
    const float* mean_prev; const float* cov_prev; const float* mp_prev=nullptr; const float* pp_prev=nullptr; const float* uh_prev=nullptr; const float* eh_prev=nullptr; float beta_prev=0;
    if(li==0){mean_prev=bmean+((size_t)b*(nb+1)+ci)*D*M;cov_prev=bcov+((size_t)b*(nb+1)+ci)*D*D;}else{mp_prev=hmp+(size_t)(li-1)*D*M;pp_prev=hpp+(size_t)(li-1)*D*D;uh_prev=huh+(size_t)(li-1)*D;eh_prev=heh+(size_t)(li-1)*M;beta_prev=hbh[li-1];}
    __syncthreads();
    if constexpr(NEED_DYNAMICS){
      for(int i=tid;i<D;i+=THREADS){float zd=0,zm=0;const float inv_di=1.0f/d[i];for(int j=0;j<D;++j){int pi=j<i?j:i;int pj=j<i?i:j;float cpv;if(li==0)cpv=cov_prev[i*D+j];else cpv=pp_prev[pi*D+pj]-beta_prev*uh_prev[i]*uh_prev[j];zd+=(gCprev[i*D+j]+gCprev[j*D+i])*cpv*inv_di;}for(int c=0;c<M;++c){float mv;if(li==0)mv=mean_prev[i*M+c];else mv=mp_prev[i*M+c]+beta_prev*uh_prev[i]*eh_prev[c];zm+=gMprev[i*M+c]*mv;}gdecay[((size_t)b*t+ti)*D+i]=zd+zm;}
      float pz=0;for(int i=tid;i<D;i+=THREADS){float inv_di=1.0f/d[i];pz+=gCprev[i*D+i]*inv_di*inv_di;}red[tid]=pz;__syncthreads();for(int off=THREADS/2;off>0;off>>=1){if(tid<off)red[tid]+=red[tid+off];__syncthreads();}if(tid==0)gprocess[(size_t)b*t+ti]=red[0];
    }
    for(int x=tid;x<D*M;x+=THREADS){int i=x/M;gM[x]=gMprev[x]*d[i];}
    for(int x=tid;x<D*D;x+=THREADS)gC[x]=gCprev[x];
    for(int i=tid;i<D;i+=THREADS)gkeys[((size_t)b*t+ti)*D+i]=gk[i];
    __syncthreads();
  }
  for(int x=tid;x<D*M;x+=THREADS)gm_state[(size_t)b*D*M+x]=gM[x];
  for(int x=tid;x<D*D;x+=THREADS)gc_state[(size_t)b*D*D+x]=gC[x];
 }
}

std::vector<torch::Tensor> forward_chunked(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,int chunks,double floor){
 TORCH_CHECK(keys.is_cuda() && values.is_cuda() && queries.is_cuda(), "fused Bayesian CUDA inputs must be CUDA tensors");
 TORCH_CHECK(keys.scalar_type()==at::kFloat && values.scalar_type()==at::kFloat && queries.scalar_type()==at::kFloat, "fused Bayesian inputs must be float32");
 TORCH_CHECK(keys.is_contiguous() && values.is_contiguous() && queries.is_contiguous() && decay.is_contiguous() && process.is_contiguous() && observation.is_contiguous() && mean0.is_contiguous() && cov0.is_contiguous(), "fused Bayesian inputs must be contiguous");
 TORCH_CHECK(chunks > 0 && keys.dim()==3 && values.dim()==3 && queries.dim()==3 && keys.size(2)==8 && values.size(2)==8 && queries.size(2)==8, "fused Bayesian kernel expects [N,T,8] tensors");
 int n=keys.size(0),t=keys.size(1),nb=(t+chunks-1)/chunks;auto out=torch::empty({n,t,8},keys.options()),mf=torch::empty_like(mean0),cf=torch::empty_like(cov0),bm=torch::empty({n,nb+1,8,8},keys.options()),bc=torch::empty({n,nb+1,8,8},keys.options());constexpr int th=32;size_t sh=(2*8*8+4*8)*sizeof(float);cudaFuncSetAttribute(chunk_forward_kernel<8,8,th>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);auto st=at::cuda::getCurrentCUDAStream();chunk_forward_kernel<8,8,th><<<n,th,sh,st>>>(keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),mean0.data_ptr<float>(),cov0.data_ptr<float>(),out.data_ptr<float>(),mf.data_ptr<float>(),cf.data_ptr<float>(),bm.data_ptr<float>(),bc.data_ptr<float>(),n,t,chunks,nb,(float)floor);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,mf,cf,bm,bc};
}

std::vector<torch::Tensor> forward_chunked_fixed(
 torch::Tensor keys,torch::Tensor values,torch::Tensor queries,torch::Tensor decay,torch::Tensor process,torch::Tensor observation,torch::Tensor mean0,torch::Tensor cov0,int chunks,double floor){
 TORCH_CHECK(keys.is_cuda() && values.is_cuda() && queries.is_cuda(), "fused Bayesian CUDA inputs must be CUDA tensors");
 // D=8 needs at most 64 cooperating lanes for the state and covariance
 // tiles.  The previous 256-thread launch left three quarters of the block
 // idle for every token (all per-row/vector loops have only 8--64 elements)
 // while still paying the same barrier cost.  Keeping 64 lanes gives the
 // covariance loops enough lanes and improves residency on SM103.
 int n=keys.size(0),t=keys.size(1),nb=(t+chunks-1)/chunks;auto out=torch::empty({n,t,8},keys.options()),mf=torch::empty_like(mean0),cf=torch::empty_like(cov0),bm=torch::empty({n,nb+1,8,8},keys.options()),bc=torch::empty({n,nb+1,8,8},keys.options());constexpr int th=64;size_t sh=(2*8*8+4*8)*sizeof(float);cudaFuncSetAttribute(chunk_forward_kernel<8,8,th,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);auto st=at::cuda::getCurrentCUDAStream();chunk_forward_kernel<8,8,th,true><<<n,th,sh,st>>>(keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),mean0.data_ptr<float>(),cov0.data_ptr<float>(),out.data_ptr<float>(),mf.data_ptr<float>(),cf.data_ptr<float>(),bm.data_ptr<float>(),bc.data_ptr<float>(),n,t,chunks,nb,(float)floor);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,mf,cf,bm,bc};
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
 // The fused replay path keeps each chunk's forward history in shared
 // memory.  At the production chunk length (64) this is ~39 KiB per block,
 // avoiding the global history tensors and the separate recompute launch.
 // Keep the existing path for unusually large chunks so dynamic shared
 // memory usage remains within the device limit.
 // The on-chip fused replay variant is retained as an opt-in experiment.
 // On SM103 its larger shared history can reduce occupancy for the full
 // dynamic backward, so the validated default remains the separate replay
 // kernels below.  Set BAYES_SSM_FUSED_REPLAY=1 to benchmark this variant.
 const char* fused_replay_env = std::getenv("BAYES_SSM_FUSED_REPLAY");
 if constexpr (!FIXED) if (chunks <= 128 && fused_replay_env && fused_replay_env[0] == '1') {
   constexpr int fth = 64;
   // Match the exact pointer layout in fused_backward_chunk_kernel.  Keeping
   // this tight matters for occupancy on SM103 (the former conservative
   // estimate reserved an extra ~9 KiB at chunk=64).
   const size_t base_floats = 552; // state/gradient tiles plus reduction
   const size_t hist_floats = (size_t)chunks * 145; // hmp+hpp+huh+hbh+heh
   const size_t shared_bytes = (base_floats + hist_floats) * sizeof(float);
   cudaFuncSetAttribute(fused_backward_chunk_kernel<8,8,fth,NEED_DYNAMICS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);
   auto st = at::cuda::getCurrentCUDAStream();
   float *gd_ptr=nullptr,*gp_ptr=nullptr,*gr_ptr=nullptr;
   if constexpr (NEED_DYNAMICS) { gd_ptr=gd.data_ptr<float>(); gp_ptr=gp.data_ptr<float>(); gr_ptr=gr.data_ptr<float>(); }
   for (int ci=nb-1; ci>=0; --ci) {
     fused_backward_chunk_kernel<8,8,fth,NEED_DYNAMICS><<<n,fth,shared_bytes,st>>>(
       keys.data_ptr<float>(),values.data_ptr<float>(),queries.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),
       bm.data_ptr<float>(),bc.data_ptr<float>(),gy.data_ptr<float>(),gk.data_ptr<float>(),gv.data_ptr<float>(),gq.data_ptr<float>(),gd_ptr,gp_ptr,gr_ptr,
       gm.data_ptr<float>(),gc.data_ptr<float>(),n,t,chunks,nb,ci,(float)floor);
     C10_CUDA_KERNEL_LAUNCH_CHECK();
   }
   if constexpr (NEED_DYNAMICS) { gm0.copy_(gm); gc0.copy_(gc); return {gk,gv,gq,gd,gp,gr,gm0,gc0}; }
   return {gk,gv,gq};
 }
 auto mp=torch::empty({n,chunks,8,8},keys.options()),pp=torch::empty_like(mp);
 auto uh=torch::empty({n,chunks,8},keys.options()),bh=torch::empty({n,chunks},keys.options());
 auto eh=torch::empty({n,chunks,8},keys.options());
 float *gd_ptr=nullptr,*gp_ptr=nullptr,*gr_ptr=nullptr;
 if constexpr (NEED_DYNAMICS) {
   gd_ptr=gd.data_ptr<float>(); gp_ptr=gp.data_ptr<float>(); gr_ptr=gr.data_ptr<float>();
 }
 // The D=8 state has only 64 mean/covariance entries.  A 64-thread block
 // covers those loops and the scalar reduction; 256 threads left 75% of the
 // block idle while paying the same barriers, reducing SM103 residency.
 constexpr int th=FIXED ? 64 : 32;
 size_t sh=(FIXED ? 3*8*8+3*8+th : 4*8*8+3*8+th)*sizeof(float);
 // Dynamic replay no longer allocates a shared covariance-prediction tile;
 // pp is written directly to global history before the matvec.  The fixed
 // specialization has always used the smaller 2*D*D footprint as well.
 cudaFuncSetAttribute(recompute_chunk_kernel<8,8,th,FIXED>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(2*8*8+4*8)*sizeof(float));
 cudaFuncSetAttribute(backward_chunk_kernel<8,8,th,NEED_DYNAMICS,FIXED>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
 auto st=at::cuda::getCurrentCUDAStream();
 for(int ci=nb-1;ci>=0;--ci){
   recompute_chunk_kernel<8,8,th,FIXED><<<n,th,(2*8*8+4*8)*sizeof(float),st>>>(
     keys.data_ptr<float>(),values.data_ptr<float>(),decay.data_ptr<float>(),process.data_ptr<float>(),observation.data_ptr<float>(),
     bm.data_ptr<float>(),bc.data_ptr<float>(),mp.data_ptr<float>(),pp.data_ptr<float>(),uh.data_ptr<float>(),bh.data_ptr<float>(),eh.data_ptr<float>(),n,t,chunks,nb,ci,(float)floor);
   backward_chunk_kernel<8,8,th,NEED_DYNAMICS,FIXED><<<n,th,sh,st>>>(
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
