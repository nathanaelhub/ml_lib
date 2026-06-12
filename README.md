# ml_lib — a tiny neural network library in pure C

[![CI](https://github.com/nathanaelhub/ml_lib/actions/workflows/ci.yml/badge.svg)](https://github.com/nathanaelhub/ml_lib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A dependency-free machine learning library written entirely in C (C11), built
on a single flat, row-major `Matrix` type. It implements the linear algebra,
activation functions, and **full backpropagation** needed to train a
multi-layer perceptron — with explicit loops and raw pointer arithmetic rather
than any external BLAS or ML framework.

The goal is clarity: every gradient is hand-derived, and a built-in
**gradient check** proves the backprop math is correct to ~1e-11.

## Highlights

- **Zero dependencies** — only the C standard library (`libm` for `exp`/`tanh`).
- **Flat `Matrix` type** — `rows`, `cols`, and one contiguous `double *data`
  block; element `(r, c)` lives at `data[r * cols + c]`.
- **Core linear algebra** — allocation/teardown, `mat_mul`, `mat_add`,
  `mat_transpose`, scaled add, random init.
- **Activations** — sigmoid, tanh, ReLU, each with its derivative.
- **Two models**
  - single sigmoid unit (logistic regression),
  - stackable `Network` MLP trained with full backpropagation.
- **Mini-batch SGD** — shuffled mini-batches with gradient accumulation.
- **Adam optimizer** — per-parameter adaptive moments with bias correction.
- **CSV dataset loader** — load numeric feature/label rows; per-sample
  access is zero-copy (a matrix row *is* a column vector).
- **Model save / load** — portable text format; exact (bit-identical) round-trip.
- **Gradient checking** — analytic gradients vs. central finite differences.
- **Memory-safe** — every allocation has a clean teardown; double-frees are
  safe; verified leak-free (valgrind / sanitizers / MSVC CRT heap).

## Layout

| File              | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `ml_lib.h`        | Public API                                                     |
| `ml_lib.c`        | Implementation (explicit loops + pointer arithmetic)           |
| `main.c`          | Five self-checking demos (OR, XOR, grad check, CSV batch, Adam)|
| `data/xor2d.csv`  | 300-point 2D nonlinear classification set for the batch demo   |
| `Makefile`        | `all` / `run` / `memcheck` / `asan` / `clean` (gcc + valgrind) |
| `CMakeLists.txt`  | Portable build (MSVC and gcc/clang) with `ctest` smoke test    |

## Build & run

### Make (Linux / macOS / MSYS2)

```sh
make run        # build and run the demos
make memcheck   # run under valgrind (expects zero leaks)
make asan       # build with AddressSanitizer + UBSan and run
```

### CMake (cross-platform)

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### MSVC (Windows, from a Developer Command Prompt)

```bat
cl /W4 /O2 ml_lib.c main.c /Fe:ml_demo.exe
ml_demo.exe
```

## What the demo shows

```
=== Demo 1: logistic regression (1 unit) on OR ===
  0 OR 0 -> 0.0020 (class 0) OK   ... accuracy: 4/4

=== Demo 2: 2-4-1 MLP on XOR (full backprop) ===
  0 XOR 0 -> 0.0028 (class 0) OK
  0 XOR 1 -> 0.9953 (class 1) OK
  1 XOR 0 -> 0.9961 (class 1) OK
  1 XOR 1 -> 0.0050 (class 0) OK   ... accuracy: 4/4

=== Demo 3: gradient check ===
  max |analytic - numerical| = 4.4e-11  (tolerance 1e-05)  PASS

=== Demo 4: mini-batch training on a CSV dataset ===
  loaded 300 samples (2 features, 1 outputs), batch size 16
  epoch  199   mean loss = 0.001485
  train accuracy: 300/300 (100.0%)

=== Demo 5: Adam optimizer + model save/load ===
  epoch   99   mean loss = 0.000475
  Adam train accuracy: 300/300 (100.0%)
  save/load round-trip: max prediction diff = 0.0e+00  PASS
```

XOR is **not** linearly separable, so a single unit provably cannot learn it —
the 2-4-1 hidden layer (tanh → sigmoid) is what makes it work. Demos 4 and 5
load a 300-point 2D nonlinear set from `data/xor2d.csv` and train a 2-8-1 net to
100% accuracy — with mini-batch SGD, then with Adam (which reaches a lower loss
in half the epochs). Demo 5 also saves the trained model, reloads it, and
confirms the predictions are bit-identical. The demo program returns a non-zero
exit code if any check fails, so it doubles as a smoke test under `ctest`.

## API sketch

```c
/* linear algebra */
Matrix mat_alloc(size_t rows, size_t cols);
void   mat_free(Matrix *m);
int    mat_mul(Matrix *out, const Matrix *a, const Matrix *b);
int    mat_transpose(Matrix *out, const Matrix *a);

/* network */
Network net_alloc(const size_t *sizes, const Activation *acts, size_t num_layers);
const Matrix *net_forward(Network *net, const Matrix *x);
double net_train_sample(Network *net, const Matrix *x, const Matrix *target, double lr);
double net_gradient_check(Network *net, const Matrix *x, const Matrix *target, double eps);
void   net_free(Network *net);

/* dataset + mini-batch training */
Dataset dataset_load_csv(const char *path, size_t n_features, size_t n_outputs, int skip_header);
double  net_train_epoch(Network *net, const Dataset *d, size_t batch_size, double lr);
void    dataset_free(Dataset *d);

/* Adam optimizer + persistence */
Optimizer adam_default(double lr);
double    net_train_epoch_adam(Network *net, const Dataset *d, size_t batch_size, Optimizer *opt);
int       net_save(const Network *net, const char *path);
Network   net_load(const char *path);
```

Build a 2-4-1 network, train it, and tear it down:

```c
const size_t     sizes[] = { 2, 4, 1 };
const Activation acts[]  = { ACT_TANH, ACT_SIGMOID };
Network net = net_alloc(sizes, acts, 2);

for (int epoch = 0; epoch < 20000; ++epoch)
    for (int s = 0; s < num_samples; ++s)
        net_train_sample(&net, &x[s], &target[s], 0.5);

net_free(&net);
```

## Roadmap

- [x] Mini-batch training (shuffled batches with gradient accumulation)
- [x] CSV / dataset loader for problems larger than a truth table
- [x] Optimizers beyond plain SGD (Adam)
- [x] Save / load trained weights
- [ ] Vectorized batch forward/backward (one `mat_mul` per layer per batch)
- [ ] Multi-class output (softmax + cross-entropy)

## License

MIT
