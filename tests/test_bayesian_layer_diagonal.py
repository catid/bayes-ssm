from __future__ import annotations

import torch

from bayes_ssm.bayesian_layer import (
    bayesian_layer,
    make_rotation_dynamics,
    make_rotation_process_covariances,
)


def test_rotation_transition_matches_dense_and_gradients() -> None:
    torch.manual_seed(46)
    batch, steps, dim, value_dim = 2, 6, 4, 3
    base = [
        torch.randn(batch, steps, dim, dtype=torch.float64),
        torch.randn(batch, steps, value_dim, dtype=torch.float64),
        torch.randn(batch, steps, dim, dtype=torch.float64),
        torch.rand(batch, steps, dim // 2, dtype=torch.float64) * 0.25 + 0.7,
        torch.randn(batch, steps, dim // 2, dtype=torch.float64) * 0.2,
        torch.rand(batch, steps, dim // 2, dtype=torch.float64) * 0.2 + 0.05,
    ]
    observation = torch.full((batch, steps), 0.3, dtype=torch.float64)

    def run(specialized: bool):
        keys, values, queries, radius, angle, process = [
            value.detach().clone().requires_grad_() for value in base
        ]
        if specialized:
            outputs = bayesian_layer(
                keys, values, queries, rotation_radii=radius,
                rotation_angles=angle, process_variances=process,
                observation_variances=observation, chunk_length=2, optimized=True,
            )
        else:
            outputs = bayesian_layer(
                keys, values, queries,
                transitions=make_rotation_dynamics(torch.log(radius), angle),
                process_covariances=make_rotation_process_covariances(process),
                observation_variances=observation, chunk_length=2, optimized=False,
            )
        loss = sum(value.square().mean() for value in outputs)
        grads = torch.autograd.grad(loss, (keys, values, queries, radius, angle, process))
        return outputs, grads

    dense, dense_grads = run(False)
    specialized, specialized_grads = run(True)
    for expected, actual in zip(dense, specialized, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
    for expected, actual in zip(dense_grads, specialized_grads, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)


def test_diagonal_transition_exact_dense_covariance_matches_dense() -> None:
    torch.manual_seed(44)
    batch, steps, dim, value_dim = 2, 7, 4, 3
    keys = torch.randn(batch, steps, dim, dtype=torch.float64)
    values = torch.randn(batch, steps, value_dim, dtype=torch.float64)
    queries = torch.randn(batch, steps, dim, dtype=torch.float64)
    diagonal = 0.8 + 0.15 * torch.rand(batch, steps, dim, dtype=torch.float64)
    process = 0.1 + 0.2 * torch.rand(batch, steps, dtype=torch.float64)
    observation = 0.2 + torch.rand(batch, steps, dtype=torch.float64)
    dense = bayesian_layer(
        keys, values, queries, transitions=torch.diag_embed(diagonal),
        process_variances=process, observation_variances=observation,
        chunk_length=3, optimized=False,
    )
    specialized = bayesian_layer(
        keys, values, queries, transition_diagonals=diagonal,
        process_variances=process, observation_variances=observation,
        chunk_length=3, optimized=True,
    )
    for expected, actual in zip(dense, specialized, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-12, atol=1e-12)


def test_diagonal_transition_gradients_and_replay_match_dense() -> None:
    torch.manual_seed(45)
    batch, steps, dim, value_dim = 1, 6, 3, 2
    base = [
        torch.randn(batch, steps, dim, dtype=torch.float64),
        torch.randn(batch, steps, value_dim, dtype=torch.float64),
        torch.randn(batch, steps, dim, dtype=torch.float64),
        0.8 + 0.15 * torch.rand(batch, steps, dim, dtype=torch.float64),
    ]
    process = torch.full((batch, steps), 0.1, dtype=torch.float64)
    observation = torch.full((batch, steps), 0.2, dtype=torch.float64)

    def run(kind: str):
        keys, values, queries, diagonal = [value.detach().clone().requires_grad_() for value in base]
        if kind == "dense":
            outputs = bayesian_layer(
                keys, values, queries, transitions=torch.diag_embed(diagonal),
                process_variances=process, observation_variances=observation,
                chunk_length=2, optimized=False,
            )
        elif kind == "specialized":
            outputs = bayesian_layer(
                keys, values, queries, transition_diagonals=diagonal,
                process_variances=process, observation_variances=observation,
                chunk_length=2, optimized=True,
            )
        else:
            outputs = bayesian_layer(
                keys, values, queries, transition_diagonals=diagonal,
                process_variances=process, observation_variances=observation,
                chunk_length=2, optimized=True, replay_ssm=True,
            )
        loss = sum(value.square().mean() for value in outputs)
        grads = torch.autograd.grad(loss, (keys, values, queries, diagonal))
        return outputs, grads

    dense, dense_grads = run("dense")
    specialized, specialized_grads = run("specialized")
    replayed, replayed_grads = run("replay")
    for expected, actual in zip(dense, specialized, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
    for expected, actual in zip(dense, replayed, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
    for expected, actual in zip(dense_grads, specialized_grads, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
    for expected, actual in zip(dense_grads, replayed_grads, strict=True):
        torch.testing.assert_close(expected, actual, rtol=1e-10, atol=1e-11)
