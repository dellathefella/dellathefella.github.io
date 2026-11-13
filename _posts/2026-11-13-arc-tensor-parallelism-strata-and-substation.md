---
layout: post
title: "Arc tensor parallelism in Strata and building Substation"
date: 2026-11-13 11:27:43 -0500
categories: projects
---

Two projects have been taking most of my time lately: bringing tensor parallelism to
Strata's SYCL engine on Intel Arc hardware, and building Substation, a local-first AI
compute orchestrator for macOS. Both are open source on GitHub.

## Tensor parallelism on Intel Arc with Strata

Strata is a local inference engine for running models like Qwen3.8-Flash-Next on your own
hardware. I added SYCL tensor parallelism to it, so a single model can be split across
multiple Intel Arc GPUs with `--tp 2..4`.

The work landed on the [`tp-sycl` branch](https://github.com/dellathefella/Strata/tree/tp-sycl)
and covers the full stack:

- **Shard table** — query and GDN heads split across ranks, dense projections split on the
  output or contraction axis, KV heads replicated when there are fewer of them than ranks,
  and MoE experts replicated per rank.
- **All-reduce** — a host-staged collective with a device-side handshake. Two ranks use the
  original peer-add; three or four ranks use a rank-0 gather/broadcast so every rank reads a
  bit-identical sum.
- **Shard-aware loader** — canonical and native dense weight uploads sized from the split
  axis, with contraction splits keeping the same block range inside every row.
- **Per-rank shapes** — GDN buffers, QSA query heads, sessions and verifiers all carve the
  rank's own shapes.
- **Serve wiring** — ranks 1..N−1 are built from their own shards, sessions, expert-cache
  replicas and verifiers; windows run in lockstep and rank 0 samples and commits the answer.

TP=2 and TP=4 both start, load the model, replicate the expert cache and capture windows.
Measured on the Arc box with the Unsloth UD-IQ4_XS pack at 32K context:

| Configuration | Decode tok/s | Prompt tok/s (66 tokens) |
|---|---:|---:|
| 2× Arc Pro B60, TP=2 | 8.8 | 18.3 |
| 4× Arc Pro B60, TP=4 | 7.2 | 16.4 |

The remaining blocker is the captured-graph all-reduce path — the standalone collective
passes on two B60s, but the engine segfaults after 16 reduces in the prompt window. The PR
is a draft while that is sorted out. Expert parallelism (each rank owning a disjoint expert
set) is the next step after correctness.

## Building Substation

[Substation](https://github.com/dellathefella/substation) is a local-first AI compute
orchestrator for macOS. One Mac — or an optional Mac cluster — automatically loads and runs
the right model for every request, orchestrated by a small triage model on the head node.

- **Auto-routing** — every request is triaged and sent to the right model tier without
  picking a model by hand.
- **Model management** — download, cache, hot-load and unload GGUF models on demand.
- **Cluster-aware** — a controller on one Mac and node agents on others; the controller
  places engines where there is RAM/VRAM room and can split layers over Thunderbolt RPC.
- **Continuous batching** — llama.cpp engines run `--parallel N` slots so concurrent
  requests are batched in-flight.
- **OpenAI-compatible API** — point any OpenAI client at `http://<host>:8800/v1`.
- **Web console and menu-bar app** — live insight into engines, models, nodes, memory and
  logs, with a single macOS app that supervises the daemon.

Both projects are open source — check out the
[Strata `tp-sycl` branch](https://github.com/dellathefella/Strata/tree/tp-sycl) and the
[Substation repository](https://github.com/dellathefella/substation) if you want to follow
along.
