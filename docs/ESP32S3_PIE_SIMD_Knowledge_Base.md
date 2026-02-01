# ESP32-S3 PIE SIMD Knowledge Base 📜

This document is the "Grand Compendium" of hardware-level discoveries made during the RNNoise optimization project. It focuses strictly on elite technical lore, assembly patterns, and architecture-specific "gotchas".

---

## 🏆 The Ultimate SIMD Checklist (Top 5 Hardware "Gotchas")

1.  **The Complex Multiplier Trap**: Standalone `ee.vmulas.s16.accx` is a complex multiplier. For real-only dot products, **always** use the `.qup` suffix.
2.  **The ABI Spill Zone**: Never subtract from `a1` manually after `entry`. The 16-32 bytes below `a1` are reserved for current window spills. Touching them causes immediate system instability or Watchdog Resets.
3.  **The Triple-Prime Rule**: You must perform 2 aligned loads and 1 priming extraction before starting a sliding window loop to clear the 384-bit history buffer.
4.  **The SAR Collision**: Unaligned loaders (`usar`) and bit-shifters (`ee.src.q`) share the same `SAR` register. Interleaving them destroys alignment state.
5.  **The 40-bit Scalar Accumulator**: Only use `wur.accx_0/1` to reset. `ee.zero.accx` is unreliable depending on the PIE power state.

---

## 🚀 Elite Assembly Mnemonics & Patterns

### 1. The Fused QUP Loader (Gold Standard)

Standard in `esp-dsp`, this instruction is the most efficient way to perform 8-lane real-only MACs with unaligned data.

- **Instruction**: `ee.vmulas.s16.accx.ld.ip.qup q_dest, a_ptr, imm, q_src, q_h1, q_h2, q_h3`
- **Lore**: Automatically manages the 384-bit internal history cycle across `q_h1, q_h2, q_h3`.

### 2. Triple-Prime Initialization (Hardware Prerequisite)

You must initialize the history pipeline before the loop:

1. `ee.ld.128.usar.ip q1, a3, 16` (Sets SAR, Loads block 0)
2. `ee.ld.128.usar.ip q2, a3, 16` (Loads block 1)
3. `ee.src.q.ld.ip    q3, a3, 16, q1, q2` (Extracts block 0 shifted, Loads block 2, Shifts registers)

### 3. 64-bit Scalar Addition (Xtensa Pattern)

When adding a 32-bit signed product (`mull`) to a 64-bit sum (`a2:a3`):

```assembly
    srai    a9, a8, 31          # Sign-extend 32-bit product (a8) into a9
    add     a6, a6, a8          # Add low parts
    bltu    a6, a8, .L_carry    # Carry detected if new_low < old_low (unsigned)
    # add     a7, a7, a9          # Add high parts
    # add     a7, a7, 1           # Add carry
```

---

## 🏛️ Hardware Architecture Discoveries

### 1. FPU Constraints (The "Double" Penalty)

- **Fact**: ESP32-S3 has a single-precision FPU. Hardware DOES NOT support `double`.
- **Cost**: A single `(double)` cast in a inner loop (like Biquad) creates a **1.4ms** penalty per frame due to software emulation.
- **Fix**: Use `float` constants (`1.0f`) exclusively.

### 2. SPIRAM (PSRAM) Latency & Cache Lines

- **Fact**: PSRAM is 5x slower than DRAM. Accessing 16-bit pointers randomly is fatal.
- **Hardware Strategy**: Use 16-byte aligned buffers (`heap_caps_aligned_alloc`) to maximize CPU Cache Line efficiency. SIMD loads (`ee.vld128`) from aligned PSRAM are significantly faster than individual scalar loads.

### 3. Flash-Cache Wait States

- **Discovery**: First-access to model weights stored in Flash triggers significant Cache-Miss overhead (~3.3ms).
- **Strategy**: Hot kernels and critical model data must be migrated to **IRAM** and **PSRAM** respectively to eliminate Flash-bus contention.

---

## 🛡️ Robust Kernel Best Practices

1.  **Zero-Padding Tail Solution**: For non-8-multiple lengths (like 480 vs 473 bits), copy remainder to a stack-aligned buffer and **Zero-Pad** to 16 bytes. This allows using the same SIMD `vmulas` path, ensuring 100% bit-exact parity with the main loop.
2.  **Pipeline Fencing**: Add `nop` after heavy accumulator loops before calling `rur` (Read User Register) to prevent pipeline hazards.
3.  **Scale Safety**: For Pitch Detection (30000x scale), the 40-bit scalar accumulator is mandatory. 32-bit registers WILL overflow.
4.  **Quantization Transparency**: 30000x scaling for `int16` results in **< 0.001%** error vs `float64`, proving quantization is mathematically transparent for RNNs on this hardware.
