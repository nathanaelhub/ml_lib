#ifndef ML_LIB_H
#define ML_LIB_H

#include <stddef.h>

/* ------------------------------------------------------------------ *
 *  ml_lib - a tiny machine learning library written entirely in C.
 *
 *  Everything is built on a single, flat, row-major Matrix type.
 *  The library deliberately avoids any external dependencies so it
 *  can be dropped into embedded / teaching projects unchanged.
 * ------------------------------------------------------------------ */

/* Row-major matrix. `data` is a single contiguous block of length
 * rows*cols. Element (r, c) lives at data[r * cols + c]. */
typedef struct {
    size_t  rows;
    size_t  cols;
    double *data;
} Matrix;

/* ----------------------------- core ------------------------------- */

/* Allocate a rows x cols matrix. Contents are zero-initialised.
 * Returns a Matrix whose `data` is NULL only if allocation failed. */
Matrix mat_alloc(size_t rows, size_t cols);

/* Release a matrix's storage and reset it to an empty state so that a
 * double-free is impossible. Safe to call on an already-freed matrix. */
void mat_free(Matrix *m);

/* Convenience accessors (bounds-unchecked, used in hot loops). */
double  mat_get(const Matrix *m, size_t r, size_t c);
void    mat_set(Matrix *m, size_t r, size_t c, double v);

/* Set every element to `v`. */
void mat_fill(Matrix *m, double v);

/* Fill with uniform random values in [low, high]. Used for weight init. */
void mat_rand(Matrix *m, double low, double high);

/* Pretty-print a matrix (with a label) to stdout. */
void mat_print(const Matrix *m, const char *name);

/* --------------------------- arithmetic --------------------------- */

/* out = a * b   (matrix product). Dimensions must conform:
 * a is (n x k), b is (k x m), out is (n x m). Returns 0 on success,
 * -1 on a dimension mismatch. Uses explicit loops + pointer arithmetic. */
int mat_mul(Matrix *out, const Matrix *a, const Matrix *b);

/* out = a + b   (element-wise). Returns 0 / -1 like mat_mul. */
int mat_add(Matrix *out, const Matrix *a, const Matrix *b);

/* in-place: dst += src, scaled by `scale`. (dst = dst + scale*src) */
int mat_add_scaled(Matrix *dst, const Matrix *src, double scale);

/* out = a^T. out must be (a->cols x a->rows) and must not alias a. */
int mat_transpose(Matrix *out, const Matrix *a);

/* --------------------------- activation --------------------------- */

double sigmoid(double x);
double sigmoid_prime_from_output(double y);   /* y = sigmoid(x) -> y*(1-y) */

/* Apply sigmoid element-wise: out[i] = sigmoid(in[i]). out may alias in. */
int mat_sigmoid(Matrix *out, const Matrix *in);

/* Selectable activation functions for network layers. */
typedef enum {
    ACT_SIGMOID = 0,
    ACT_TANH,
    ACT_RELU
} Activation;

/* Scalar activation and its derivative w.r.t. the pre-activation z. */
double act_apply(double x, Activation a);
double act_prime(double z, Activation a);

/* ------------------------- neural network ------------------------- */

/* A single fully-connected layer:  a = activation(W * x + b)
 *
 * W is (outputs x inputs), b is (outputs x 1). On its own this models
 * logistic regression; stacked inside a Network it becomes one layer of
 * an MLP. The z/a/delta/gW/gb matrices are training caches: they are
 * left empty ({0,0,NULL}) for the bare single-layer API and allocated
 * by net_alloc() when the layer is part of a Network. */
typedef struct {
    size_t      inputs;
    size_t      outputs;
    Matrix      W;          /* weights          (outputs x inputs) */
    Matrix      b;          /* bias             (outputs x 1)      */
    Activation  act;        /* activation function                */
    Matrix      z;          /* cache: pre-activation W*x+b         */
    Matrix      a;          /* cache: activation act(z)            */
    Matrix      delta;      /* cache: error term dL/dz             */
    Matrix      gW;         /* cache: weight gradient dL/dW        */
    Matrix      gb;         /* cache: bias gradient   dL/db        */
} Layer;

/* Bare single-layer API (logistic regression). Caches are not allocated.
 * layer_free is the "clean teardown" used by the leak-verification step. */
Layer layer_alloc(size_t inputs, size_t outputs);
void  layer_free(Layer *l);

/* Forward pass for a single sample using a sigmoid activation.
 *   x   : (inputs x 1) input column
 *   out : (outputs x 1) receives sigmoid(W*x + b)
 * Returns 0 on success, -1 on a dimension mismatch. */
int layer_forward(const Layer *l, const Matrix *x, Matrix *out);

/* One gradient-descent update for a single sample (binary cross-entropy
 * with a sigmoid output, so the delta simplifies to (pred - target)).
 * Returns the summed squared error for this sample (for monitoring). */
double layer_backward(Layer *l, const Matrix *x, const Matrix *pred,
                      const Matrix *target, double lr);

/* ---------------------- multi-layer network ----------------------- */

/* A feed-forward stack of Layers, trained with full backpropagation. */
typedef struct {
    size_t  num_layers;     /* number of weight layers */
    Layer  *layers;
} Network;

/* Build a network.
 *   sizes : array of (num_layers+1) sizes: [n_in, h1, ..., h_k, n_out]
 *   acts  : array of num_layers activations, one per layer
 * Returns a Network whose `layers` is NULL on allocation failure. */
Network net_alloc(const size_t *sizes, const Activation *acts,
                  size_t num_layers);
void    net_free(Network *net);

/* Forward pass for one sample; caches z/a in every layer. Returns a
 * pointer to the output activation (last layer's `a`), or NULL on a
 * dimension mismatch. */
const Matrix *net_forward(Network *net, const Matrix *x);

/* 1/2 * sum((a_out - target)^2) using the activations cached by the most
 * recent net_forward call. */
double net_loss(const Network *net, const Matrix *target);

/* Backpropagation for one sample: fills gW/gb in every layer. Requires a
 * prior net_forward(net, x) with the same x. Does NOT update weights. */
int net_backprop(Network *net, const Matrix *x, const Matrix *target);

/* Apply the gradients from net_backprop: W -= lr*gW, b -= lr*gb. */
void net_step(Network *net, double lr);

/* Convenience: forward + backprop + step on one sample. Returns the loss
 * measured before the weight update. */
double net_train_sample(Network *net, const Matrix *x, const Matrix *target,
                        double lr);

/* --------------------------- gradient check ----------------------- */

/* Verify backprop against numerical gradients for a single sample by
 * perturbing every weight & bias by +/-eps and comparing the
 * finite-difference gradient to the analytic one. Returns the largest
 * absolute difference found (expect ~1e-7 or smaller when correct). */
double net_gradient_check(Network *net, const Matrix *x,
                          const Matrix *target, double eps);

#endif /* ML_LIB_H */
