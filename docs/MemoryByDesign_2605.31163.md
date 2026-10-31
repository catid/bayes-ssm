---
title: "Memory by Design: Bayesian Layers for Sequence Modeling"
authors: "Matthew Dowling; Hyungju Jeon; Cristina Savin; Il Memming Park"
arxiv: "2605.31163v5"
submitted: "2026-05-29; revised 2026-10-14"
---

# Memory by Design: Bayesian Layers for Sequence Modeling

This is a hand-cleaned technical extraction of arXiv:2605.31163v5. The paper defines recurrent sequence layers by first specifying a probabilistic *design model* for a latent memory and then deriving the recurrent update by Bayesian filtering. Its main instance is the Bayesian Layer (BL), a Kalman update over a matrix-valued associative memory.

## 1. Design-model framework

A recurrent layer has state $h_t=f(h_{t-1},x_t)$. A design model introduces a latent memory $S_t$, an initial density $\mu_0(S_0)$, and nonnegative input-dependent pairwise potentials $\Psi_t(S_{t-1},S_t;x_t)$:

$$
\pi_t(S_{0:t};x_{1:t}) = Z_t(x_{1:t})^{-1}\mu_0(S_0)\prod_{s=1}^t\Psi_s(S_{s-1},S_s;x_s).
$$

Bayesian filtering gives

$$
\mu_t(S_t)\propto \int \Psi_t(S_{t-1},S_t;x_t)\mu_{t-1}(S_{t-1})\,dS_{t-1}.
$$

A readout $g(S_t,x_t)$ is reported as its posterior mean, $y_t=\mathbb E_{\mu_t}[g(S_t,x_t)]$, which is Bayes-optimal under squared error. The belief parameters needed by future updates and reads are the fixed-size recurrent state.

## 2. Bayesian Layer design model

For key dimension $D$ and value dimension $m$, the latent memory is $S_t\in\mathbb R^{D\times m}$. Input-dependent functions produce key $k_t\in\mathbb R^D$, value $v_t\in\mathbb R^m$, transition $A_t\in\mathbb R^{D\times D}$, process-noise scale $\ell_t>0$, and observation-noise scale $r_t>0$. The initial belief has independent Gaussian columns with mean $M_0=0$ and covariance $P_0=\sigma_0^2I$.

$$
\Psi_t=\exp\left[-\frac{\|S_t-A_tS_{t-1}\|_F^2}{2\ell_t^2}
-\frac{\|v_t-S_t^\top k_t\|_2^2}{2r_t^2}\right].
$$

Equivalently, $S_t\mid S_{t-1}\sim\mathcal N(A_tS_{t-1},\ell_t^2I)$ columnwise and $v_t\mid S_t\sim\mathcal N(S_t^\top k_t,r_t^2I)$, evaluated at the supplied $(k_t,v_t)$. Filtering preserves independent Gaussian columns with shared covariance:

$$
\mu_t(S_t)=\prod_{j=1}^m\mathcal N(S_t^{(j)};M_t^{(j)},P_t),
\qquad \operatorname{vec}(S_t)\sim\mathcal N(\operatorname{vec}(M_t),I_m\otimes P_t).
$$

The query $q_t=q(x_t)\in\mathbb R^D$ defines the linear readout $g(S_t,x_t)=S_t^\top q_t$ and therefore

$$y_t=M_t^\top q_t,\qquad \operatorname{Var}(S_t^\top q_t)=q_t^\top P_tq_t.
$$

## 3. Core Bayesian Layer recurrence

Given $(M_{t-1},P_{t-1})$, first predict through the dynamics and then condition on $(k_t,v_t)$:

$$
\begin{aligned}
\bar P_t &= A_tP_{t-1}A_t^\top+\ell_t^2I, \\
 u_t &= \bar P_tk_t,\\
 \beta_t&=(r_t^2+k_t^\top u_t)^{-1},\\
 \bar M_t&=A_tM_{t-1},\\
 M_t&=(I-\beta_tu_tk_t^\top)\bar M_t+\beta_tu_tv_t^\top\\
     &=\bar M_t+\beta_tu_t(v_t-\bar M_t^\top k_t)^\top,\\
 P_t&=\bar P_t-\beta_tu_tu_t^\top,\\
 y_t&=M_t^\top q_t.
\end{aligned}
$$

$u_t$ is the uncertainty-warped write direction. The denominator is positive for $r_t>0$ and $P_t\succeq0$; implementation floors it at $10^{-6}$. The covariance is dense and shared across all $m$ value columns, while the mean update costs $O(Dm)$.

### Connections recovered by changing the design model/filter

Resetting the predicted covariance to $\bar P_t=\lambda_tI$ gives the Delta-rule family:

$$
\eta_t=\frac{\lambda_t}{r_t^2+\lambda_t\|k_t\|^2},\qquad
M_t=(I-\eta_tk_tk_t^\top)A_tM_{t-1}+\eta_tk_tv_t^\top.
$$

A separate latent-input model uses $\phi_t\sim\mathcal N(0,\lambda_tI)$, $v_t\mid\phi_t\sim\mathcal N(\phi_t,r_t^2I)$, and $S_t=A_tS_{t-1}+k_t\phi_t^\top$. Its exact posterior mean is additive:

$$
M_t=A_tM_{t-1}+\omega_tk_tv_t^\top,qquad
\omega_t=\frac{\lambda_t}{\lambda_t+r_t^2}.
$$

With $A_t=I$ this is linear attention; scalar decay gives RetNet, diagonal decay gives GLA, and the equivalent scalar form gives Mamba-2/SSD. The distinction is between changing the underlying probabilistic design model and approximating its filter.

## 4. Efficient chunked implementation

The exact dense recurrence is sequential in time. The paper implements chunks of length $L$ with Triton kernels. Each chunk receives $(M_0,P_0)$ and returns the exit belief; chunk-boundary states are retained for backward, with optional checkpointing/replay.

Use structured dynamics: diagonal $A_t$ or damped 2D rotations

$$
A_t^{(j)}=\rho_{t,j}
\begin{bmatrix}\cos\theta_{t,j}&-\sin\theta_{t,j}\\
\sin\theta_{t,j}&\cos\theta_{t,j}\end{bmatrix},
\qquad 0<\rho_{t,j}\le1.
$$

For rotational blocks, keys and queries are embedded as $(k_{t,j},0)$ and $(q_{t,j},0)$; process noise uses $Q_t^{(j)}=\ell_t^2\begin{bmatrix}1&0\\0&0\end{bmatrix}$. Ordinary diagonal dynamics use $Q_t=\ell_t^2I$. Noise variances use softplus with a positive floor. One covariance is maintained per head and shared across that head's value columns.

For a chunk with entry covariance $P_0$, define forward products $\Phi_{t\leftarrow s}=A_t\cdots A_{s+1}$ and $\Phi_{s\leftarrow s}=I$:

$$
\Pi_t=\Phi_{t\leftarrow0}P_0\Phi_{t\leftarrow0}^\top+\sum_{s=1}^t\Phi_{t\leftarrow s}Q_s\Phi_{t\leftarrow s}^\top.
$$

Let $w_{t,s}=\Phi_{t\leftarrow s}u_s$. Unrolling prior rank-one covariance downdates gives

$$
\bar P_t=\Pi_t-\sum_{s<t}\beta_sw_{t,s}w_{t,s}^\top,
$$

so the next gain can be computed without materializing a dense covariance at every position:

$$
\begin{aligned}
u_t&=\Pi_tk_t-\sum_{s<t}\beta_sw_{t,s}(w_{t,s}^\top k_t),\\
\beta_t&=(r_t^2+k_t^\top u_t)^{-1}.
\end{aligned}
$$

A sequential loop inside the chunk retains preceding directions and gains; value-column tiles carry the mean recurrence and emit $y_t=M_t^\top q_t$. At chunk exit, collect $W=[\Phi_{L\leftarrow1}u_1,\ldots,\Phi_{L\leftarrow L}u_L]$ and return

$$
P_L=\Pi_L-W\operatorname{diag}(\beta_1,\ldots,\beta_L)W^\top.
$$

State storage per head is $O(Dm+D^2)$. For chunk length $L$, the stated costs are $O(LD^2)$ for covariance entry/exit, $O(L^2D)$ for gain histories, and $O(LDm)$ for mean/readout; over sequence length $T$, fixed $L$ gives linear scaling, approximately $O(T(D^2+LD+Dm))$. Recurrent states and chunk boundaries use FP32. Forward transition products use cumulative log-radii and angles (avoiding explicit inverse transitions); the norm is bounded by one under the parameterization. A custom backward differentiates covariance products, gains, means/readouts, and exit covariance.

## 5. Reproduction protocols

### Controlled-overlap toy and learned recall

- Toy: six one-hot associations in $D=16$, $k_A=e_1$, $k_B=0.98e_1+\sqrt{1-0.98^2}e_2$, four orthogonal keys; two random initialization rounds, then ten B writes and ten A writes. BL uses $P_0=3I$, $A=I$, $\ell^2=r^2=0.05$; average all $900$ initial order classes.
- Learned controlled recall: $K=8$ target/distractor pairs in address dimension 16, overlaps sampled $\rho_i\sim U(0.60,0.80)$ for training, $n_b=4$ target writes and $n_d\in\{1,2,4,8\}$ distractor writes; query targets afterward. Evaluation uses $\rho=0.98$ and $n_d$ through 64, or $n_d=64$ with overlaps up to 0.98. Two pre-norm RMSNorm mixer/MLP blocks, width 64, four heads with key/value dimension 16, SwiGLU width 128, no positional embeddings. AdamW lr $3\times10^{-4}$, $(\beta_1,\beta_2)=(0.9,0.999)$, weight decay $10^{-4}$, clip 1, batch 256, 2,500 updates, three seeds.
- The bounded runner exposes both parts of this protocol: `--overlap-sweep` adds an $n_d=64$ sweep over configurable bounds (default $\rho=0.60,\ldots,0.98$), and `--seeds 123 234 345` runs independent seeds and emits population-standard-deviation aggregates for the Figure 3 curves, matching §E.3.1. (The paper's summary tables use sample standard deviations.) For a short GPU smoke use:

  ```bash
  python -m memory_by_design.experiments.controlled_recall \
    --device cuda:0 --steps 100 --batch-size 16 --eval-batch 64 \
    --optimized --overlap-sweep --seeds 123 234 345
  ```
- BL uses identity dynamics, fixed $r^2=0.05$, learned input-dependent process variance $\ell_t^2>0$, and initial covariance $3I$ per sequence. The diagonal variant drops off-diagonal $P$ entries after each update. Keys and queries are supplied addresses so learned projections cannot alter overlap.

### MQAR

Use the canonical Zoology MQAR generator, vocabulary 8,192. Training: 100k examples at $(L,K)=(64,4)$ and 20k each at $(128,8),(256,16),(256,32),(256,64)$; evaluate 1k examples at $(64,4),(64,8),(64,16),(128,32),(256,64),(512,128),(1024,256)$. Widths are 64/128/256, two mixer blocks, tied embeddings, two heads with key/value width $d/2$, no MLP. BL uses identity dynamics and learned input-dependent process/observation noise (process variance initialized 0.1), GDN-like length-4 causal convolutions, SiLU Q/K/V projections, L2-normalized Q/K, and RMS-normalized readout. AdamW $(0.9,0.95)$, $\epsilon=10^{-8}$, wd 0.1, clip 1, batch 256, up to 32 epochs, epoch cosine to zero, no warmup; early stop at validation accuracy 0.99. Pilots select lr from $\{3e\!-4,6e\!-4,1e\!-3\}$ (paper uses $10^{-3}$ except MesaNet $6e\!-4$); seeds 123/234/345, mean and sample standard deviation. The runner's `--mixed --pool-scale 1 --epochs 32 --seeds 123 234 345` command exposes this full pool and schedule; its bounded results are labelled separately.

The GDN short-convolution implementation uses the upstream default `conv_bias=False`; the constructors expose this as an override for ablations.

### WikiText-103

Use official WikiText-103, the 32k-BPE `fla-hub/gla-1.3B-100B` tokenizer, BOS and packed 2,048-token windows. Width 512, four mixer/MLP blocks, tied embeddings, 520,028,160 tokens, 7,935 updates of 65,536 tokens, seeds 0/1/2, paired streams. AdamW $(0.9,0.95)$, $\epsilon=10^{-8}$, wd 0.01, clip 1; 20M-token warmup then cosine to 0.1× peak; peak lr $10^{-3}$; MLP dropout 0.4. Evaluate LM perplexity and associative recall by distance bins 1–64, 65–256, 257–1,024 tokens.

The runner exposes this token-based schedule through `--warmup-tokens`, `--total-tokens`, and `--cosine-floor`; bounded runs retain the same schedule shape even when their local batch size is smaller.

### Distillation into pretrained Gated DeltaNet

Start from `linear-moe-hub/Gated-Deltanet-340M` (24 layers, hidden 1,024, four heads of 256). Replace mixers at indices `{5,11,17,23}` with BL rotational dynamics, no short convolution; each BL head has 512 real state coordinates and a shared 512×512 covariance. Initialize the rotational BL with unit prior covariance, process variance 0.05, and observation variance 1.0 before training the noise parameters. Fit each replacement to frozen GDN outputs with

$$
\mathcal L=\frac{\operatorname{MSE}(Y^{BL},Y^{GDN})}{\operatorname{Var}(Y^{GDN})+10^{-6}}
+0.05\frac1B\sum_b\left(1-\cos(y_b^{BL},y_b^{GDN})\right).
$$

Freeze MLPs and remaining backbone. Train each replacement independently for 1,526 updates / 100,007,936 tokens on 2,048-token SlimPajama sequences (global 65,536 tokens), AdamW lr $5e\!-4$, $(0.9,0.999)$, no wd, clip 1, 25 warmup updates, cosine to 0.1× peak, mixed precision, three seeds. Continued-GDN control uses 763 updates of 131,072 tokens, lr $3e\!-4$, $(0.9,0.95)$, 50 warmup updates, same token budget and no wd. Evaluate six RULER needle-in-a-haystack tasks at context lengths 2k/4k/8k/16k, 200 examples each, cached prefill and greedy generation up to 128 tokens.

## 6. Implementation caveats

- Dense $P_t$ is the defining cost and scales quadratically in key dimension; diagonal/structured approximations change the model.
- The paper's notation switches between $M_0,P_0$ and the initial design-model covariance $\sigma_0^2I$; use the explicitly specified initial covariance for each experiment.
- Input-dependent $A_t$, $k_t$, $q_t$, $v_t$, $r_t$, and $\ell_t$ are treated as supplied functions; exact projection architectures differ by experiment (controlled recall, MQAR, WikiText, distillation).
- Rotation blocks use a real-coordinate embedding and process noise only in the first coordinate; ordinary diagonal dynamics use isotropic noise. This is easy to implement incorrectly.
- The text says chunk length $L$ while the model and experimental sequence lengths also use $L$; keep chunk length and total sequence length distinct.
- Full custom backward/checkpointing and Triton tiling are required for the reported efficiency. A straightforward Python/PyTorch loop reproduces equations but not throughput.
- Process/observation noise positivity uses softplus floors; innovation denominators are floored at $10^{-6}$ in floating point.
- Results are averages over three seeds with sample standard deviations; reproduce paired data ordering/streams where specified.
