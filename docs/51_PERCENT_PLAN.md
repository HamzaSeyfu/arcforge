# ARCForge 51% Plan

## Why 51% is now a rational milestone

ARC Prize officially verified Poetiq at 54% on the ARC-AGI-2 semi-private set,
improving Gemini 3 Pro from a ~31% baseline to 54% through a refinement harness.
That is much more relevant to our target than public-eval-only 95% claims.

The lesson is not 'use one better prompt'. The verified system repeatedly:
- asks a model to write an executable transform;
- executes it on every training example;
- computes exact and soft feedback;
- feeds the best failed attempts back into later iterations;
- keeps multiple experts/seeds;
- groups identical test outputs and votes with diversity-first ordering.

Poetiq's open config uses up to 10 iterations, keeps up to 5 prior solutions as
feedback, and supports 1/2/8 parallel experts. Its feedback includes exact
shape/cell mismatches and a soft cell-accuracy score.

Confluence reaches 97.92% on public eval with a more brute-force version of the
same broad idea: 12 independent agents per test input, up to 10 refinement loops,
exact execution verification, then majority voting over generated grids.

Athanor adds the missing anti-overfit ingredient: a second, independent reviewer
can reject a program even after it fits all training examples.

Squeeze-Evolve adds a resource-allocation insight: spend expensive reasoning on
high-diversity disagreements; use cheap aggregation when candidates already agree.

## ARCForge architecture

Near-term target: 51% Kaggle-style capability.

Pipeline:

    ARC task
       |
       +--> Qwen/NVARC direct grid candidates
       |
       +--> deterministic ARCForge specialists
       |
       +--> local code-model program search
                  |
             initial programs
                  |
            execute on train
                  |
        exact fits / near misses
                  |
          diff + soft feedback
                  |
              mutations
                  |
          exact-fit population
                  |
          generalization gate
                  |
                  +--------------------+
                                       |
                              combined candidate pool
                                       |
                              support + diversity
                                       |
                               attempt_1 / attempt_2

## What has been implemented

The branch research/51pct-agentic-search now contains:
- a restricted executable Python verifier;
- exact-train and soft cell-level scoring;
- Poetiq-style diff feedback;
- behavior-based program deduplication;
- parent selection that preserves failure-pattern diversity;
- iterative mutation/refinement engine;
- exact-fit candidate aggregation by support and source diversity;
- a pluggable local text-model generator;
- a lazy local Qwen runtime for Kaggle-attached weights;
- a cheap structural anti-overfit reviewer;
- a public-eval benchmark script.

## Model choice

Qwen3-Coder-30B-A3B-Instruct is a promising offline program generator because
it is Apache-2.0, has 30.5B total parameters but only 3.3B activated per token,
and is explicitly optimized for agentic coding. It is not yet assumed to fit
our Kaggle runtime in the final configuration; this must be benchmarked.

A smaller model may win once throughput is included. The engine intentionally
does not depend on a specific model so we can swap Qwen3-Coder, Qwen3-8B,
GPT-OSS, or another attached open-weight model without rewriting search logic.

## Kill gates

1. On a 10-task slice, refinement must produce exact-fit programs on tasks where
   the initial generation failed. Otherwise the model/search prompt is wrong.
2. On 30 tasks, agentic search must add unique oracle solves over Qwen + current
   ARCForge experts. Otherwise do not scale it to all 120.
3. Before a Kaggle submission, combined candidate oracle should exceed 55%, not
   merely 51%, because the selector will fail to recover every oracle hit.
4. Search cost must fit the Kaggle offline wall-clock budget after routing.

## Next experiment

Attach a local code-capable model in Kaggle and run scripts/benchmark_agentic_search.py
on 10 public-eval tasks first. Measure:
- exact-fit program rate;
- refinement gain from generation 0 to later generations;
- number of distinct exact-fit test outputs;
- runtime per task;
- unique oracle gain over the existing 59/172 combined pool.
