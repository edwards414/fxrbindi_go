"""Measure single-position checkpoint latency for each app strength level."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import pgx

from gozero.mcts import batch_of_one, load_ckpt, make_search_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()

    env, net, params, _ = load_ckpt(args.ckpt)
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    state = batch_of_one(env.init(init_key))
    results = {}

    for simulations in (0, 32, 128):
        search = make_search_fn(env, net, num_simulations=simulations)
        key, warmup_key = jax.random.split(key)
        action, value = search(params, warmup_key, state)
        action.block_until_ready()
        value.block_until_ready()

        samples = []
        for _ in range(args.repetitions):
            key, sample_key = jax.random.split(key)
            started = time.perf_counter()
            action, value = search(params, sample_key, state)
            action.block_until_ready()
            value.block_until_ready()
            samples.append((time.perf_counter() - started) * 1000)
        results[str(simulations)] = round(statistics.median(samples))

    print(json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
