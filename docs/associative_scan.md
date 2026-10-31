# Associative scan design

For identity dynamics and zero process noise, each observation contributes an additive precision and natural-parameter update:

\[
\Lambda_t = \Lambda_0 + \sum_{s\le t} k_s k_s^T / r_s^2,
\qquad
N_t = N_0 + \sum_{s\le t} k_s v_s^T / r_s^2.
\]

The posterior mean is recovered from `Lambda_t M_t = N_t`. The CUDA implementation performs the prefix sums, symmetrizes the precision prefix for FP32 stability, and uses batched Cholesky solves/inverses. This preserves dense covariance; no diagonal approximation is used.

For learned transitions or process noise, the covariance Riccati update is non-associative. Those cases use the exact dense ReplaySSM recurrence instead.
