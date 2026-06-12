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
    Matrix      mW, vW;     /* Adam 1st/2nd moment estimates for W */
    Matrix      mb, vb;     /* Adam 1st/2nd moment estimates for b */
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

/* Backpropagation for one sample: overwrites gW/gb in every layer with
 * that sample's gradient (it zeroes the accumulators first). Requires a
 * prior net_forward(net, x) with the same x. Does NOT update weights. */
int net_backprop(Network *net, const Matrix *x, const Matrix *target);

/* Zero every layer's gradient accumulators (gW, gb). Call before summing
 * a mini-batch's gradients with net_backprop_accum. */
void net_zero_grad(Network *net);

/* Add one sample's gradient into the accumulators (gW += ..., gb += ...)
 * instead of overwriting. Requires a prior net_forward(net, x). This is
 * the building block of mini-batch training. */
int net_backprop_accum(Network *net, const Matrix *x, const Matrix *target);

/* Apply the accumulated gradients: W -= lr*gW, b -= lr*gb. For a mini-batch
 * pass lr scaled by 1/batch_size to average the summed gradient. */
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

/* ----------------------------- dataset ---------------------------- */

/* A dataset stored as two row-major matrices: each row is one sample.
 * Because a matrix row is contiguous, row i of X is already laid out as a
 * (n_features x 1) column vector, so per-sample views need no copy. */
typedef struct {
    size_t  n_samples;
    size_t  n_features;
    size_t  n_outputs;
    Matrix  X;      /* (n_samples x n_features) feature rows */
    Matrix  Y;      /* (n_samples x n_outputs)  label rows   */
} Dataset;

Dataset dataset_alloc(size_t n_samples, size_t n_features, size_t n_outputs);
void    dataset_free(Dataset *d);

/* Load a numeric CSV where each row is n_features feature columns followed
 * by n_outputs label columns (comma/space/semicolon/tab separated). If
 * skip_header is non-zero the first line is ignored. Returns a Dataset
 * whose X.data is NULL if the file could not be opened or held no rows. */
Dataset dataset_load_csv(const char *path, size_t n_features,
                         size_t n_outputs, int skip_header);

/* Non-owning column-vector views into sample i. Do NOT mat_free these;
 * they point into the dataset's own storage. */
Matrix dataset_input(const Dataset *d, size_t i);    /* (n_features x 1) */
Matrix dataset_target(const Dataset *d, size_t i);   /* (n_outputs  x 1) */

/* ------------------------ mini-batch training --------------------- */

/* One epoch of mini-batch SGD: shuffles the samples, and for each
 * mini-batch accumulates the per-sample gradients and applies a single
 * averaged update. Returns the mean loss over the epoch (-1.0 on error). */
double net_train_epoch(Network *net, const Dataset *d, size_t batch_size,
                       double lr);

/* --------------------------- Adam optimizer ----------------------- */

/* Adam (Kingma & Ba, 2014): per-parameter adaptive learning rates from
 * running estimates of the gradient's 1st and 2nd moments. The moment
 * buffers live in each Layer; this struct holds the hyper-parameters and
 * the shared timestep used for bias correction. */
typedef struct {
    double lr;
    double beta1;       /* 1st-moment decay (typ. 0.9)    */
    double beta2;       /* 2nd-moment decay (typ. 0.999)  */
    double eps;         /* numerical floor  (typ. 1e-8)   */
    long   t;           /* timestep, incremented per step */
} Optimizer;

/* An Optimizer with the usual Adam defaults (0.9 / 0.999 / 1e-8, t=0). */
Optimizer adam_default(double lr);

/* Apply one Adam update from the accumulated gradients (gW/gb), which are
 * treated as a sum over `batch_size` samples so the mean gradient is used.
 * Increments opt->t. */
void net_step_adam(Network *net, Optimizer *opt, size_t batch_size);

/* One epoch of shuffled mini-batch training using Adam instead of plain
 * SGD. Returns the mean loss over the epoch (-1.0 on error). */
double net_train_epoch_adam(Network *net, const Dataset *d, size_t batch_size,
                            Optimizer *opt);

/* --------------------------- persistence -------------------------- */

/* Save / load a network's architecture and weights as a portable text
 * file. net_save returns 0 on success, -1 on an I/O error. net_load builds
 * a fresh network (free it with net_free); on failure its `layers` is NULL.
 * Round-trips exactly: reloaded weights are bit-identical to the saved ones. */
int     net_save(const Network *net, const char *path);
Network net_load(const char *path);

#endif /* ML_LIB_H */
