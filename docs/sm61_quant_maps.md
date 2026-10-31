# Quantization maps for GTX 1080 Ti (sm_61, 11 GB)

Measured with the bundled `llama.cpp-sm_61` ggml on one GTX 1080 Ti (CUDA 12.8, `--flash-attn on`, fixed seed), after the
first (cold) run had warmed the file cache. Speeds are steady-state seconds per sampling step. Pascal has no fast FP16, so
quantized weights (Q8_0 and INT8) are *faster* than F16/BF16 here, not only smaller.

## Per-component map

Each component of a model is quantized on its own. The type is what to pass to `-M convert --type ...`
(or `--tensor-type-rules` for a single tensor group); `int8` is the ComfyUI int8 tensorwise/convrot format.

| Family (tested model) | Diffusion model | Text encoder | VAE | Start with | Step time | Diffusion weights |
|---|---|---|---|---|---|---|
| SD 1.5 (Lemix) | f16 | f16 | f16 | `--flash-attn on` | 0.48 s @512² | 1.6 GB |
| SDXL (Illustrious) | **q8_0** | f16 | f16 | `--flash-attn on` | 2.23 s @1024² (f16: 3.19) | 2.9 GB (f16 4.9) |
| SDXL, low memory | q4_0 | f16 | f16 | same | 2.19 s @1024² | 1.8 GB |
| Pony / SDXL derivatives (Hassaku) | q8_0 | f16 | f16 | same | 2.23 s @1024² | 2.9 GB |
| Z-Image Turbo | **int8** | Qwen3-4B q8_0 or bf16 | bf16 | `--offload-to-cpu --flash-attn on` | 4.97 s @1024² (q8_0 5.61, q4_0 5.44, q4_K 5.83, bf16 10.0) | 6.0 GB (q4: 3.3) |
| Krea2 (Kroma, INT8 source) | int8 (keep the source) | Qwen3-VL-4B fp8 | bf16 | `--offload-to-cpu --flash-attn on` | 6.4 s @512², 10.1 s @1024² | 12.5 GB |
| Qwen-Image 1.0 (20B) | **q4_K** | Qwen2.5-VL-7B q8_0/bf16 | bf16 | `--offload-to-cpu --flash-attn on --flow-shift 3` | 8.1 s @768² with CFG (bf16: 24 s @512²) | 11.5 GB (bf16 41) |
| Wan 2.2 TI2V 5B | q8_0 | umt5-xxl q4_K_M | wan2.2_vae | `-M vid_gen --offload-to-cpu --flash-attn on` | ≈2 s/step @480×320, 17 frames | 5.4 GB |

Notes:

* Qwen-Image does not fit in 11 GB even at 4 bits; `--offload-to-cpu` streams layers, so the host RAM/PCIe bandwidth
  sets the speed. A faster file system matters only for the first run.
* Do not convert INT8 sources to GGUF with an sd-cli that predates the INT8 metadata keys: the resulting `*-Q4_K.gguf`
  files (for example the Krea2 Kroma one) load as plain I8 and produce noise. Re-convert them with the current build.
* INT8 weights of the DiT linear layers are chosen by the tensor-type rules; embeddings, norms, biases and convolutions stay
  in their original type.
* Z-Image INT8 produced by `--type int8` from BF16 reaches 26.2 dB PSNR against the BF16 image (Q8_0: 25.9, Q4_K: 16.7,
  Q4_0: 15.5), so 4-bit variants change the image noticeably on few-step models; use them only when memory demands it.

## INT4

A dedicated INT4 weight type was not added: with per-group scales it is Q4_0, which already runs on `dp4a` (the nibbles are
unpacked to int8 and `__dp4a` does the 4-way dot product; `dp2a` only helps 16-bit x 8-bit operands). Per-row INT4 without
groups is about four times worse in weight error than Q4_0 (NMSE 3.2e-2 vs 8e-3 on Z-Image layers), and a Walsh-Hadamard
rotation before quantization improves Q4_0 by at most 16% (7.4e-3 vs 8.8e-3).
