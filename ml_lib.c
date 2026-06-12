/* Use the portable C library (fopen/strtok/strtod) without MSVC's
 * _s-variant deprecation warnings under /W4. */
#define _CRT_SECURE_NO_WARNINGS 1

#include "ml_lib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================== core ============================== */

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    /* calloc gives us zero-initialised storage in one shot. */
    m.data = (double *)calloc(rows * cols, sizeof(double));
    if (m.data == NULL) {
        /* Signal failure by zeroing the dimensions too. */
        m.rows = 0;
        m.cols = 0;
    }
    return m;
}

void mat_free(Matrix *m)
{
    if (m == NULL)
        return;
    free(m->data);      /* free(NULL) is a no-op, so double-free is safe */
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

double mat_get(const Matrix *m, size_t r, size_t c)
{
    return m->data[r * m->cols + c];
}

void mat_set(Matrix *m, size_t r, size_t c, double v)
{
    m->data[r * m->cols + c] = v;
}

void mat_fill(Matrix *m, double v)
{
    double *p   = m->data;
    double *end = m->data + m->rows * m->cols;
    while (p < end)
        *p++ = v;
}

void mat_rand(Matrix *m, double low, double high)
{
    double  span = high - low;
    double *p    = m->data;
    double *end  = m->data + m->rows * m->cols;
    while (p < end) {
        double u = (double)rand() / (double)RAND_MAX;   /* u in [0,1] */
        *p++ = low + u * span;
    }
}

void mat_print(const Matrix *m, const char *name)
{
    printf("%s = [ %zu x %zu ]\n", name ? name : "M", m->rows, m->cols);
    for (size_t r = 0; r < m->rows; ++r) {
        printf("  ");
        for (size_t c = 0; c < m->cols; ++c)
            printf("% .4f ", m->data[r * m->cols + c]);
        printf("\n");
    }
}

/* =========================== arithmetic =========================== */

int mat_mul(Matrix *out, const Matrix *a, const Matrix *b)
{
    /* (n x k) * (k x m) = (n x m) */
    if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols)
        return -1;

    const size_t n = a->rows;
    const size_t k = a->cols;
    const size_t m = b->cols;

    for (size_t i = 0; i < n; ++i) {
        const double *a_row = a->data + i * k;      /* start of row i of A */
        double       *o_row = out->data + i * m;    /* start of row i of OUT */
        for (size_t j = 0; j < m; ++j) {
            double sum = 0.0;
            /* dot product of A's row i with B's column j */
            for (size_t p = 0; p < k; ++p)
                sum += a_row[p] * b->data[p * m + j];
            o_row[j] = sum;
        }
    }
    return 0;
}

int mat_add(Matrix *out, const Matrix *a, const Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols ||
        out->rows != a->rows || out->cols != a->cols)
        return -1;

    size_t        n   = a->rows * a->cols;
    const double *pa  = a->data;
    const double *pb  = b->data;
    double       *po  = out->data;
    for (size_t i = 0; i < n; ++i)
        po[i] = pa[i] + pb[i];
    return 0;
}

int mat_add_scaled(Matrix *dst, const Matrix *src, double scale)
{
    if (dst->rows != src->rows || dst->cols != src->cols)
        return -1;

    size_t        n  = dst->rows * dst->cols;
    double       *pd = dst->data;
    const double *ps = src->data;
    for (size_t i = 0; i < n; ++i)
        pd[i] += scale * ps[i];
    return 0;
}

int mat_transpose(Matrix *out, const Matrix *a)
{
    if (out->rows != a->cols || out->cols != a->rows)
        return -1;

    for (size_t r = 0; r < a->rows; ++r)
        for (size_t c = 0; c < a->cols; ++c)
            out->data[c * a->rows + r] = a->data[r * a->cols + c];
    return 0;
}

/* =========================== activation =========================== */

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime_from_output(double y)
{
    return y * (1.0 - y);
}

int mat_sigmoid(Matrix *out, const Matrix *in)
{
    if (out->rows != in->rows || out->cols != in->cols)
        return -1;

    size_t        n  = in->rows * in->cols;
    const double *pi = in->data;
    double       *po = out->data;
    for (size_t i = 0; i < n; ++i)
        po[i] = sigmoid(pi[i]);
    return 0;
}

double act_apply(double x, Activation a)
{
    switch (a) {
    case ACT_TANH: return tanh(x);
    case ACT_RELU: return x > 0.0 ? x : 0.0;
    case ACT_SIGMOID:
    default:       return sigmoid(x);
    }
}

double act_prime(double z, Activation a)
{
    switch (a) {
    case ACT_TANH: { double t = tanh(z); return 1.0 - t * t; }
    case ACT_RELU: return z > 0.0 ? 1.0 : 0.0;
    case ACT_SIGMOID:
    default:       { double s = sigmoid(z); return s * (1.0 - s); }
    }
}

size_t mat_argmax(const Matrix *m)
{
    size_t n    = m->rows * m->cols;
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
        if (m->data[i] > m->data[best])
            best = i;
    return best;
}

/* Softmax over a layer's pre-activations, written into its activation
 * cache. Max-shifted for numerical stability: a_k = e^(z_k - max) / sum. */
static void layer_softmax(Layer *l)
{
    double m = l->z.data[0];
    for (size_t o = 1; o < l->outputs; ++o)
        if (l->z.data[o] > m)
            m = l->z.data[o];

    double sum = 0.0;
    for (size_t o = 0; o < l->outputs; ++o) {
        double e = exp(l->z.data[o] - m);
        l->a.data[o] = e;
        sum += e;
    }
    for (size_t o = 0; o < l->outputs; ++o)
        l->a.data[o] /= sum;
}

/* ===================== single layer (bare API) ==================== */

Layer layer_alloc(size_t inputs, size_t outputs)
{
    const Matrix empty = { 0, 0, NULL };
    Layer l;
    l.inputs  = inputs;
    l.outputs = outputs;
    l.W   = mat_alloc(outputs, inputs);
    l.b   = mat_alloc(outputs, 1);
    l.act = ACT_SIGMOID;
    /* Training caches stay empty until a Network allocates them. */
    l.z = empty; l.a = empty; l.delta = empty; l.gW = empty; l.gb = empty;
    l.mW = empty; l.vW = empty; l.mb = empty; l.vb = empty;

    /* Small symmetric random weights break the symmetry between units;
     * the bias starts at zero. */
    mat_rand(&l.W, -1.0, 1.0);
    mat_fill(&l.b, 0.0);
    return l;
}

void layer_free(Layer *l)
{
    if (l == NULL)
        return;
    mat_free(&l->W);
    mat_free(&l->b);
    mat_free(&l->z);
    mat_free(&l->a);
    mat_free(&l->delta);
    mat_free(&l->gW);
    mat_free(&l->gb);
    mat_free(&l->mW);
    mat_free(&l->vW);
    mat_free(&l->mb);
    mat_free(&l->vb);
    l->inputs  = 0;
    l->outputs = 0;
}

int layer_forward(const Layer *l, const Matrix *x, Matrix *out)
{
    if (x->rows != l->inputs  || x->cols != 1 ||
        out->rows != l->outputs || out->cols != 1)
        return -1;

    /* out = W * x  (outputs x 1) */
    if (mat_mul(out, &l->W, x) != 0)
        return -1;
    /* out += b */
    if (mat_add(out, out, &l->b) != 0)
        return -1;
    /* out = sigmoid(out) */
    return mat_sigmoid(out, out);
}

double layer_backward(Layer *l, const Matrix *x, const Matrix *pred,
                      const Matrix *target, double lr)
{
    /* For a sigmoid output trained with binary cross-entropy, the error
     * signal feeding the weights is simply (pred - target). We also
     * accumulate squared error here purely for reporting. */
    double sse = 0.0;

    for (size_t o = 0; o < l->outputs; ++o) {
        double p     = pred->data[o];
        double t     = target->data[o];
        double diff  = p - t;             /* dL/d(pre-activation) */
        sse += diff * diff;

        /* W[o][i] -= lr * diff * x[i]; row o of W is contiguous. */
        double *w_row = l->W.data + o * l->inputs;
        const double *xp = x->data;
        for (size_t i = 0; i < l->inputs; ++i)
            w_row[i] -= lr * diff * xp[i];

        /* Bias update (input is implicitly 1). */
        l->b.data[o] -= lr * diff;
    }

    return sse;
}

/* ====================== multi-layer network ======================= */

Network net_alloc(const size_t *sizes, const Activation *acts,
                  size_t num_layers)
{
    Network net;
    net.num_layers = num_layers;
    net.layers     = NULL;
    if (num_layers == 0)
        return net;

    net.layers = (Layer *)malloc(num_layers * sizeof(Layer));
    if (net.layers == NULL) {
        net.num_layers = 0;
        return net;
    }

    for (size_t i = 0; i < num_layers; ++i) {
        size_t in  = sizes[i];
        size_t out = sizes[i + 1];

        Layer l = layer_alloc(in, out);
        l.act   = acts[i];

        /* Allocate the training caches used by forward/backprop. */
        l.z     = mat_alloc(out, 1);
        l.a     = mat_alloc(out, 1);
        l.delta = mat_alloc(out, 1);
        l.gW    = mat_alloc(out, in);
        l.gb    = mat_alloc(out, 1);

        /* Adam moment estimates (zeroed by calloc inside mat_alloc). */
        l.mW    = mat_alloc(out, in);
        l.vW    = mat_alloc(out, in);
        l.mb    = mat_alloc(out, 1);
        l.vb    = mat_alloc(out, 1);

        /* Fan-in scaled init keeps activations in a sane range. */
        double scale = 1.0 / sqrt((double)in);
        mat_rand(&l.W, -scale, scale);

        net.layers[i] = l;
    }
    return net;
}

void net_free(Network *net)
{
    if (net == NULL)
        return;
    for (size_t i = 0; i < net->num_layers; ++i)
        layer_free(&net->layers[i]);
    free(net->layers);          /* free(NULL) safe -> idempotent teardown */
    net->layers     = NULL;
    net->num_layers = 0;
}

const Matrix *net_forward(Network *net, const Matrix *x)
{
    const Matrix *input = x;
    for (size_t i = 0; i < net->num_layers; ++i) {
        Layer *l = &net->layers[i];
        if (input->rows != l->inputs || input->cols != 1)
            return NULL;

        /* z = W*input + b */
        if (mat_mul(&l->z, &l->W, input) != 0)
            return NULL;
        if (mat_add(&l->z, &l->z, &l->b) != 0)
            return NULL;

        /* a = activation(z). Softmax couples all outputs; the rest are
         * element-wise. */
        if (l->act == ACT_SOFTMAX)
            layer_softmax(l);
        else
            for (size_t o = 0; o < l->outputs; ++o)
                l->a.data[o] = act_apply(l->z.data[o], l->act);

        input = &l->a;          /* feed this layer's output to the next */
    }
    return &net->layers[net->num_layers - 1].a;
}

double net_loss(const Network *net, const Matrix *target)
{
    const Layer *out = &net->layers[net->num_layers - 1];

    if (out->act == ACT_SOFTMAX) {
        /* Categorical cross-entropy: -sum_k t_k log(a_k). The 1e-15 floor
         * guards against log(0) for a class the net drove to ~0. */
        double ce = 0.0;
        for (size_t o = 0; o < out->outputs; ++o) {
            double a = out->a.data[o] < 1e-15 ? 1e-15 : out->a.data[o];
            ce -= target->data[o] * log(a);
        }
        return ce;
    }

    double sum = 0.0;
    for (size_t o = 0; o < out->outputs; ++o) {
        double d = out->a.data[o] - target->data[o];
        sum += d * d;
    }
    return 0.5 * sum;
}

void net_zero_grad(Network *net)
{
    for (size_t li = 0; li < net->num_layers; ++li) {
        mat_fill(&net->layers[li].gW, 0.0);
        mat_fill(&net->layers[li].gb, 0.0);
    }
}

int net_backprop_accum(Network *net, const Matrix *x, const Matrix *target)
{
    size_t L = net->num_layers;
    if (L == 0)
        return -1;

    /* Output layer delta = dL/dz. For softmax + cross-entropy this is just
     * (a - t) -- the softmax Jacobian and the CE derivative cancel. For the
     * element-wise activations (with squared-error loss) it is the same
     * (a - t) scaled by act'(z). */
    Layer *out = &net->layers[L - 1];
    for (size_t o = 0; o < out->outputs; ++o) {
        double d = out->a.data[o] - target->data[o];
        out->delta.data[o] = (out->act == ACT_SOFTMAX)
                                 ? d
                                 : d * act_prime(out->z.data[o], out->act);
    }

    /* Hidden layers, propagating delta backward.
     * li runs from L-2 down to 0 (the underflow guard `li-- > 0` stops it). */
    for (size_t li = L - 1; li-- > 0; ) {
        Layer *cur  = &net->layers[li];
        Layer *next = &net->layers[li + 1];
        /* delta_cur = (W_next^T * delta_next) .* act'(z_cur) */
        for (size_t i = 0; i < cur->outputs; ++i) {
            double sum = 0.0;
            /* column i of W_next, i.e. weights leaving unit i */
            for (size_t j = 0; j < next->outputs; ++j)
                sum += next->W.data[j * next->inputs + i] * next->delta.data[j];
            cur->delta.data[i] = sum * act_prime(cur->z.data[i], cur->act);
        }
    }

    /* Accumulate gradients:  gW_l += delta_l * a_{l-1}^T,  gb_l += delta_l
     * with a_{-1} == x (the network input). Summing across a mini-batch is
     * what makes this batch (rather than pure online) gradient descent. */
    for (size_t li = 0; li < L; ++li) {
        Layer        *l    = &net->layers[li];
        const Matrix *prev = (li == 0) ? x : &net->layers[li - 1].a;
        for (size_t o = 0; o < l->outputs; ++o) {
            double  dl    = l->delta.data[o];
            double *gwrow = l->gW.data + o * l->inputs;
            for (size_t i = 0; i < l->inputs; ++i)
                gwrow[i] += dl * prev->data[i];
            l->gb.data[o] += dl;
        }
    }
    return 0;
}

int net_backprop(Network *net, const Matrix *x, const Matrix *target)
{
    /* Single-sample gradient = zeroed accumulators + one accumulation. */
    net_zero_grad(net);
    return net_backprop_accum(net, x, target);
}

void net_step(Network *net, double lr)
{
    for (size_t li = 0; li < net->num_layers; ++li) {
        Layer *l = &net->layers[li];
        mat_add_scaled(&l->W, &l->gW, -lr);
        mat_add_scaled(&l->b, &l->gb, -lr);
    }
}

double net_train_sample(Network *net, const Matrix *x, const Matrix *target,
                        double lr)
{
    if (net_forward(net, x) == NULL)
        return -1.0;
    double loss = net_loss(net, target);    /* loss before the update */
    net_backprop(net, x, target);
    net_step(net, lr);
    return loss;
}

double net_gradient_check(Network *net, const Matrix *x,
                          const Matrix *target, double eps)
{
    /* Analytic gradients (cached in each layer's gW/gb). */
    if (net_forward(net, x) == NULL)
        return -1.0;
    net_backprop(net, x, target);

    double max_diff = 0.0;

    for (size_t li = 0; li < net->num_layers; ++li) {
        Layer *l = &net->layers[li];

        /* Each weight: numerical gradient via central difference. */
        size_t nW = l->W.rows * l->W.cols;
        for (size_t k = 0; k < nW; ++k) {
            double orig = l->W.data[k];

            l->W.data[k] = orig + eps;
            net_forward(net, x);
            double lp = net_loss(net, target);

            l->W.data[k] = orig - eps;
            net_forward(net, x);
            double lm = net_loss(net, target);

            l->W.data[k] = orig;            /* restore */

            double num  = (lp - lm) / (2.0 * eps);
            double diff = fabs(num - l->gW.data[k]);
            if (diff > max_diff)
                max_diff = diff;
        }

        /* Each bias. */
        for (size_t k = 0; k < l->b.rows; ++k) {
            double orig = l->b.data[k];

            l->b.data[k] = orig + eps;
            net_forward(net, x);
            double lp = net_loss(net, target);

            l->b.data[k] = orig - eps;
            net_forward(net, x);
            double lm = net_loss(net, target);

            l->b.data[k] = orig;            /* restore */

            double num  = (lp - lm) / (2.0 * eps);
            double diff = fabs(num - l->gb.data[k]);
            if (diff > max_diff)
                max_diff = diff;
        }
    }

    net_forward(net, x);    /* leave caches in a clean, consistent state */
    return max_diff;
}

/* ============================= dataset ============================ */

Dataset dataset_alloc(size_t n_samples, size_t n_features, size_t n_outputs)
{
    Dataset d;
    d.n_samples  = n_samples;
    d.n_features = n_features;
    d.n_outputs  = n_outputs;
    d.X = mat_alloc(n_samples, n_features);
    d.Y = mat_alloc(n_samples, n_outputs);
    return d;
}

void dataset_free(Dataset *d)
{
    if (d == NULL)
        return;
    mat_free(&d->X);
    mat_free(&d->Y);
    d->n_samples  = 0;
    d->n_features = 0;
    d->n_outputs  = 0;
}

Matrix dataset_input(const Dataset *d, size_t i)
{
    /* Row i of X is n_features contiguous doubles -> a (n_features x 1)
     * column vector that shares storage with the dataset (non-owning). */
    Matrix v;
    v.rows = d->n_features;
    v.cols = 1;
    v.data = d->X.data + i * d->n_features;
    return v;
}

Matrix dataset_target(const Dataset *d, size_t i)
{
    Matrix v;
    v.rows = d->n_outputs;
    v.cols = 1;
    v.data = d->Y.data + i * d->n_outputs;
    return v;
}

/* Return non-zero if a line is blank (only whitespace). */
static int line_is_blank(const char *s)
{
    while (*s) {
        if (*s != ' ' && *s != '\t' && *s != '\r' && *s != '\n')
            return 0;
        ++s;
    }
    return 1;
}

Dataset dataset_load_csv(const char *path, size_t n_features,
                         size_t n_outputs, int skip_header)
{
    const Dataset empty = { 0, 0, 0, { 0, 0, NULL }, { 0, 0, NULL } };
    const char   *DELIMS = ",;\t \r\n";

    FILE *f = fopen(path, "r");
    if (f == NULL)
        return empty;

    char   line[8192];
    size_t cols = n_features + n_outputs;

    /* Pass 1: count non-blank data rows. */
    size_t rows  = 0;
    int    first = 1;
    while (fgets(line, sizeof(line), f)) {
        if (first && skip_header) { first = 0; continue; }
        first = 0;
        if (!line_is_blank(line))
            ++rows;
    }
    if (rows == 0) {
        fclose(f);
        return empty;
    }

    Dataset d = dataset_alloc(rows, n_features, n_outputs);
    if (d.X.data == NULL || d.Y.data == NULL) {
        fclose(f);
        dataset_free(&d);
        return empty;
    }

    /* Pass 2: parse each row into the feature/label matrices. */
    rewind(f);
    first = 1;
    size_t r = 0;
    while (fgets(line, sizeof(line), f) && r < rows) {
        if (first && skip_header) { first = 0; continue; }
        first = 0;
        if (line_is_blank(line))
            continue;

        size_t c   = 0;
        char  *tok = strtok(line, DELIMS);
        while (tok != NULL && c < cols) {
            double val = strtod(tok, NULL);
            if (c < n_features)
                d.X.data[r * n_features + c] = val;
            else
                d.Y.data[r * n_outputs + (c - n_features)] = val;
            ++c;
            tok = strtok(NULL, DELIMS);
        }
        ++r;
    }

    fclose(f);
    return d;
}

int dataset_split(const Dataset *d, double val_frac,
                  Dataset *train, Dataset *val)
{
    if (d == NULL || train == NULL || val == NULL)
        return -1;
    if (val_frac < 0.0 || val_frac > 1.0)
        return -1;

    size_t n       = d->n_samples;
    size_t n_val   = (size_t)(val_frac * (double)n);
    size_t n_train = n - n_val;

    size_t *idx = (size_t *)malloc(n * sizeof(size_t));
    if (idx == NULL)
        return -1;
    for (size_t i = 0; i < n; ++i)
        idx[i] = i;
    for (size_t i = n; i-- > 1; ) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    *train = dataset_alloc(n_train, d->n_features, d->n_outputs);
    *val   = dataset_alloc(n_val,   d->n_features, d->n_outputs);
    if ((n_train && train->X.data == NULL) || (n_val && val->X.data == NULL)) {
        free(idx);
        dataset_free(train);
        dataset_free(val);
        return -1;
    }

    /* Copy each shuffled row into the train (first n_train) or val set. */
    for (size_t k = 0; k < n; ++k) {
        size_t   src = idx[k];
        Dataset *dst = (k < n_train) ? train : val;
        size_t   row = (k < n_train) ? k : (k - n_train);
        memcpy(dst->X.data + row * d->n_features,
               d->X.data   + src * d->n_features,
               d->n_features * sizeof(double));
        memcpy(dst->Y.data + row * d->n_outputs,
               d->Y.data   + src * d->n_outputs,
               d->n_outputs * sizeof(double));
    }

    free(idx);
    return 0;
}

/* ======================= mini-batch training ===================== */

/* Shared epoch loop. If `opt` is non-NULL the batch update uses Adam;
 * otherwise it is plain SGD with learning rate `lr`. */
static double train_epoch_impl(Network *net, const Dataset *d,
                               size_t batch_size, double lr, Optimizer *opt)
{
    size_t n = d->n_samples;
    if (n == 0 || batch_size == 0)
        return 0.0;

    size_t *idx = (size_t *)malloc(n * sizeof(size_t));
    if (idx == NULL)
        return -1.0;
    for (size_t i = 0; i < n; ++i)
        idx[i] = i;

    /* Fisher-Yates shuffle so each epoch sees a fresh sample order. */
    for (size_t i = n; i-- > 1; ) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }

    double total = 0.0;
    for (size_t start = 0; start < n; start += batch_size) {
        size_t size = (start + batch_size <= n) ? batch_size : (n - start);

        net_zero_grad(net);
        for (size_t k = 0; k < size; ++k) {
            size_t i  = idx[start + k];
            Matrix xv = dataset_input(d, i);
            Matrix tv = dataset_target(d, i);
            if (net_forward(net, &xv) == NULL) {
                free(idx);
                return -1.0;
            }
            total += net_loss(net, &tv);
            net_backprop_accum(net, &xv, &tv);
        }
        if (opt != NULL)
            net_step_adam(net, opt, size);
        else
            /* Average the summed gradient over the batch via the scaled rate. */
            net_step(net, lr / (double)size);
    }

    free(idx);
    return total / (double)n;
}

double net_train_epoch(Network *net, const Dataset *d, size_t batch_size,
                       double lr)
{
    return train_epoch_impl(net, d, batch_size, lr, NULL);
}

double net_train_epoch_adam(Network *net, const Dataset *d, size_t batch_size,
                            Optimizer *opt)
{
    return train_epoch_impl(net, d, batch_size, 0.0, opt);
}

double net_accuracy(Network *net, const Dataset *d)
{
    if (d->n_samples == 0)
        return 0.0;

    size_t correct = 0;
    for (size_t i = 0; i < d->n_samples; ++i) {
        Matrix        xv  = dataset_input(d, i);
        Matrix        tv  = dataset_target(d, i);
        const Matrix *out = net_forward(net, &xv);
        if (d->n_outputs == 1) {
            int p = out->data[0] >= 0.5 ? 1 : 0;
            int t = tv.data[0]   >= 0.5 ? 1 : 0;
            correct += (p == t);
        } else {
            correct += (mat_argmax(out) == mat_argmax(&tv));
        }
    }
    return 100.0 * (double)correct / (double)d->n_samples;
}

/* ========================= Adam optimizer ======================== */

Optimizer adam_default(double lr)
{
    Optimizer o;
    o.lr    = lr;
    o.beta1 = 0.9;
    o.beta2 = 0.999;
    o.eps   = 1e-8;
    o.t     = 0;
    return o;
}

/* Adam update for one parameter block. `gsum` is the gradient summed over
 * the batch; we scale by inv_b to get the mean. bc1/bc2 are the bias
 * corrections (1 - beta^t) shared across all parameters this step. */
static void adam_update(Matrix *p, const Matrix *gsum, Matrix *m, Matrix *v,
                        const Optimizer *opt, double inv_b,
                        double bc1, double bc2)
{
    size_t n = p->rows * p->cols;
    for (size_t i = 0; i < n; ++i) {
        double g = gsum->data[i] * inv_b;
        m->data[i] = opt->beta1 * m->data[i] + (1.0 - opt->beta1) * g;
        v->data[i] = opt->beta2 * v->data[i] + (1.0 - opt->beta2) * g * g;
        double mhat = m->data[i] / bc1;
        double vhat = v->data[i] / bc2;
        p->data[i] -= opt->lr * mhat / (sqrt(vhat) + opt->eps);
    }
}

void net_step_adam(Network *net, Optimizer *opt, size_t batch_size)
{
    double inv_b = 1.0 / (double)batch_size;
    opt->t += 1;
    double bc1 = 1.0 - pow(opt->beta1, (double)opt->t);
    double bc2 = 1.0 - pow(opt->beta2, (double)opt->t);

    for (size_t li = 0; li < net->num_layers; ++li) {
        Layer *l = &net->layers[li];
        adam_update(&l->W, &l->gW, &l->mW, &l->vW, opt, inv_b, bc1, bc2);
        adam_update(&l->b, &l->gb, &l->mb, &l->vb, opt, inv_b, bc1, bc2);
    }
}

/* ========================== persistence ========================== */

int net_save(const Network *net, const char *path)
{
    FILE *f = fopen(path, "w");
    if (f == NULL)
        return -1;

    fprintf(f, "ml_lib 1\n");
    fprintf(f, "%zu\n", net->num_layers);
    for (size_t li = 0; li < net->num_layers; ++li) {
        const Layer *l = &net->layers[li];
        fprintf(f, "%zu %zu %d\n", l->inputs, l->outputs, (int)l->act);
    }
    /* Weights then bias for each layer, full precision for exact round-trip. */
    for (size_t li = 0; li < net->num_layers; ++li) {
        const Layer *l  = &net->layers[li];
        size_t       nw = l->W.rows * l->W.cols;
        for (size_t k = 0; k < nw; ++k)
            fprintf(f, "%.17g\n", l->W.data[k]);
        for (size_t k = 0; k < l->b.rows; ++k)
            fprintf(f, "%.17g\n", l->b.data[k]);
    }

    fclose(f);
    return 0;
}

Network net_load(const char *path)
{
    Network empty;
    empty.num_layers = 0;
    empty.layers     = NULL;

    FILE *f = fopen(path, "r");
    if (f == NULL)
        return empty;

    char magic[32];
    int  ver = 0;
    if (fscanf(f, "%31s %d", magic, &ver) != 2 || strcmp(magic, "ml_lib") != 0) {
        fclose(f);
        return empty;
    }

    size_t L = 0;
    if (fscanf(f, "%zu", &L) != 1 || L == 0) {
        fclose(f);
        return empty;
    }

    size_t     *sizes = (size_t *)malloc((L + 1) * sizeof(size_t));
    Activation *acts  = (Activation *)malloc(L * sizeof(Activation));
    if (sizes == NULL || acts == NULL) {
        free(sizes); free(acts); fclose(f);
        return empty;
    }

    int ok = 1;
    for (size_t i = 0; i < L; ++i) {
        size_t in, out;
        int    act;
        if (fscanf(f, "%zu %zu %d", &in, &out, &act) != 3) { ok = 0; break; }
        if (i == 0)
            sizes[0] = in;
        else if (in != sizes[i]) { ok = 0; break; }    /* layers must chain */
        sizes[i + 1] = out;
        acts[i]      = (Activation)act;
    }
    if (!ok) {
        free(sizes); free(acts); fclose(f);
        return empty;
    }

    Network net = net_alloc(sizes, acts, L);
    free(sizes);
    free(acts);
    if (net.layers == NULL) {
        fclose(f);
        return empty;
    }

    for (size_t li = 0; li < L && ok; ++li) {
        Layer *l  = &net.layers[li];
        size_t nw = l->W.rows * l->W.cols;
        for (size_t k = 0; k < nw; ++k)
            if (fscanf(f, "%lf", &l->W.data[k]) != 1) { ok = 0; break; }
        for (size_t k = 0; k < l->b.rows && ok; ++k)
            if (fscanf(f, "%lf", &l->b.data[k]) != 1) { ok = 0; break; }
    }

    fclose(f);
    if (!ok) {
        net_free(&net);
        return empty;
    }
    return net;
}
