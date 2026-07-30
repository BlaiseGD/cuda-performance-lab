# Layer Normalization

A CUDA implementation of Layer Normalization, built and optimized in stages, with each version profiled using Nsight Compute.

- **Naive** — one thread per token, serial reduction
- **Block Reduction** — one block per token, reduction in shared memory
- **Warp Reduction** — warp-level primitives replace the shared-memory reduction
- **Fused** — warp reduction + shared-memory reduction combined

Runtime went from 419.36 μs (naive) to as low as 11.97 μs (warp reduction) — a 35× speedup. Full writeup and Nsight Compute analysis included in the repo.

## Hardware

GPU: RTX 2060 Super (`sm_75`)
