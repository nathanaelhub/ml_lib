# ml_lib — a tiny neural network library in pure C

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
- **Gradient checking** — analytic gradients vs. central finite differences.
- **Memory-safe** — every allocation has a clean teardown; double-frees are
  safe; verified leak-free (valgrind / sanitizers / MSVC CRT heap).

## Layout

| File             | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `ml_lib.h`       | Public API                                                     |
| `ml_lib.c`       | Implementation (explicit loops + pointer arithmetic)           |
| `main.c`         | Three self-checking demos (OR, XOR, gradient check)            |
| `Makefile`       | `all` / `run` / `memcheck` / `asan` / `clean` (gcc + valgrind) |
| `CMakeLists.txt` | Portable build (MSVC and gcc/clang) with `ctest` smoke test    |

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
```

XOR is **not** linearly separable, so a single unit provably cannot learn it —
the 2-4-1 hidden layer (tanh → sigmoid) is what makes it work. The demo program
returns a non-zero exit code if any check fails, so it doubles as a smoke test
under `ctest`.

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

- Mini-batch training (turn `mat_mul` into the real workhorse)
- CSV / dataset loader for problems larger than a truth table
- Optimizers beyond plain SGD (momentum, Adam)
- Save / load trained weights

## License

MIT
