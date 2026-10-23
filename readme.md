# TinyTensorA64

TinyTensorA64 is a small C++ neural-network inference runtime built to explore low-level CPU performance optimization, with a particular focus on **AArch64 and Arm NEON SIMD**.
The project implements a lightweight tensor and neural-network.
It supports basic tensor operations and a small feed-forward neural network while providing several different matrix multiplication implementations for performance experimentation.

---

## Matrix Multiplication Optimizations

The original matrix multiplication implementation used a standard triple loop
where the resulting matrix would be traversed while an inner loop
would be used to sum the neccesary values.
This first implementation focused entirely on correctness before performance optimization.

| Size | Runtime |
|---|---:|
| 128x128 | 5928µs average |
| 256x256 | 47428µs average |
| 512x512 | 462950µs average |
| 1024x1024 | 7545608µs average |

---

### Loop Reordering

The first optimization changed the order in which matrix elements were accessed.
instead of accessing the second matrix column by column, the loop walks down the row before dropping columns.
By doing this, we can make better use of cache lines and adhere to the principal of spatial locality.

| Size | Runtime |
|---|---:|
| 128x128 | 3905 average  |
| 256x256 | 29206µs average |
| 512x512 | 248029µs average |
| 1024x1024 | 1911844µs average |

---

### Cache Blocking / Tiling

Matrix multiplication was then split into smaller tiles.
Instead of processing the complete matrices at once, the computation works on smaller regions.
The goal here was to reuse useful matrix data while it is still in CPU cache.
Several block sizes were tested because there is no universally optimal tile size.
Small blocks improve cache locality but introduce more loop overhead.
Large blocks reduce blocking overhead but increase the working set and may reduce cache effectiveness.

| Matrix Size | Block 16 | Block 32 | Block 64 |
|---|---:|---:|---:|
| 64×64 | 377 µs | 387 µs | 362 µs |
| 128×128 | 3,315 µs | 3,227 µs | 3,171 µs |
| 256×256 | 25,363 µs | 25,600 µs | 25,946 µs |
| 512×512 | 211,683 µs | 206,807 µs | 210,594 µs |
| 1024×1024 | 1,729,686 µs | 1,711,282 µs | 1,686,869 µs |

---

### Multithreading

Matrix multiplication was also parallelized using `std::thread`.
The work was divided between threads so that each worker owned separate result rows or tiles.
On the Ryzen 5 5600, larger matrices benefited significantly from additional worker threads.
Small matrices benefited less because thread creation and scheduling overhead become large relative to the amount of computation.

| Matrix Size | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|---|---:|---:|---:|---:|
| 64×64 | 729 µs | 511 µs | 589 µs | 644 µs |
| 128×128 | 5,371 µs | 2,931 µs | 1,992 µs | 1,756 µs |
| 256×256 | 38,203 µs | 24,702 µs | 13,823 µs** | 13,846 µs |
| 512×512 | 314,798 µs | 169,076 µs | 97,548 µs | 61,568 µs |
| 1024×1024 | 2,628,475 µs | 1,396,110 µs | 711,661 µs | 420,167 µs |


---

### AArch64 NEON SIMD

The runtime was then tested on real AArch64 hardware using an Arm Neoverse-N1 VM.
Arm NEON SIMD allows a single CPU instruction to operate on several values simultaneously.
For a `float`, a 128-bit NEON vector contains four 32-bit floating-point values.
Instead of updating one matrix value at a time, the NEON kernel processes four neighbouring columns.

The implementation uses NEON intrinsics including:

```cpp
vdupq_n_f32()
vld1q_f32()
vfmaq_f32()
vst1q_f32()
```

This produced a substantial improvement over the portable matrix multiplication implementation.

| Matrix Size | Loop Optimized | NEON SIMD | Speedup |
|---|---:|---:|---:|
| 64×64 | 379 µs | **207 µs** | 1.83× |
| 128×128 | 2,872 µs | **1,447 µs** | 1.98× |
| 256×256 | 23,306 µs | **11,105 µs** | 2.10× |
| 512×512 | 186,997 µs | **126,813 µs** | 1.47× |
| 1024×1024 | 1,479,178 µs | **737,952 µs** | 2.00× |

---

### Blocked + NEON

Cache blocking and NEON were also combined.
This implementation attempts to gain both:

- improved cache locality from tiling
- four-float SIMD processing from NEON

#### Blocked Matrix Multiplication

| Matrix Size | Block 16 | Block 32 | Block 64 | Block 128 | Block 256 |
|---|---:|---:|---:|---:|---:|
| 64×64 | 398 µs | **328 µs** | 338 µs | 351 µs | 329 µs |
| 128×128 | 2,825 µs | 2,625 µs | 2,541 µs | **2,509 µs** | 2,511 µs |
| 256×256 | 23,240 µs | 21,258 µs | 20,682 µs | 20,402 µs | **20,279 µs** |
| 512×512 | 198,259 µs | 170,559 µs | 177,078 µs | 164,263 µs | **162,459 µs** |
| 1024×1024 | 1,525,468 µs | 1,370,774 µs | **1,337,514 µs** | 1,338,843 µs | 1,367,530 µs |

#### Blocked + NEON SIMD

| Matrix Size | Block 16 | Block 32 | Block 64 | Block 128 | Block 256 |
|---|---:|---:|---:|---:|---:|
| 64×64 | 218 µs | 181 µs | 197 µs | 191 µs | **173 µs** |
| 128×128 | 1,615 µs | 1,446 µs | 1,371 µs | **1,348 µs** | 1,353 µs |
| 256×256 | 13,241 µs | 11,916 µs | 11,242 µs | 11,249 µs | **10,865 µs** |
| 512×512 | 117,538 µs | 97,653 µs | 97,318 µs | 91,742 µs | **90,333 µs** |
| 1024×1024 | 988,534 µs | 851,812 µs | 805,965 µs | **779,732 µs** | 801,958 µs |

From the results it can be seen that generally tiling with NEON 
performs better than without SIMD. However at larger sizes like
1024x1024 a dropoff in performance was observed and even performed
worse than the SIMD implementation without tiling.
I can conclude here that the repeated calculations and 
the cost of tiling generated too much overhead and outweighed
the cache savings.

---


## Hardware / Environment

Benchmarks were collected on two systems.

### x86-64

```text
CPU: AMD Ryzen 5 5600
Cores: 6
Threads: 12
Architecture: x86-64
```

This system was primarily used for:

- loop-order experiments
- cache blocking
- multithreading
- parallel blocked matrix multiplication

### AArch64

```text
CPU: Arm Neoverse-N1
Architecture: AArch64
vCPUs: 1
Threads per core: 1
NEON / Advanced SIMD: supported
```

This system was primarily used for:

- portable Arm performance
- cache-blocking experiments
- GCC vectorization analysis
- NEON SIMD development
- NEON benchmarking

---
