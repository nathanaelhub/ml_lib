#include "ml_lib.h"

#include <stdio.h>
#include <stdlib.h>
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

        /* a = activation(z) */
        for (size_t o = 0; o < l->outputs; ++o)
            l->a.data[o] = act_apply(l->z.data[o], l->act);

        input = &l->a;          /* feed this layer's output to the next */
    }
    return &net->layers[net->num_layers - 1].a;
}

double net_loss(const Network *net, const Matrix *target)
{
    const Layer *out = &net->layers[net->num_layers - 1];
    double sum = 0.0;
    for (size_t o = 0; o < out->outputs; ++o) {
        double d = out->a.data[o] - target->data[o];
        sum += d * d;
    }
    return 0.5 * sum;
}

int net_backprop(Network *net, const Matrix *x, const Matrix *target)
{
    size_t L = net->num_layers;
    if (L == 0)
        return -1;

    /* Output layer:  delta = (a - t) * act'(z) */
    Layer *out = &net->layers[L - 1];
    for (size_t o = 0; o < out->outputs; ++o) {
        double d = out->a.data[o] - target->data[o];
        out->delta.data[o] = d * act_prime(out->z.data[o], out->act);
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

    /* Gradients:  gW_l = delta_l * a_{l-1}^T,  gb_l = delta_l
     * with a_{-1} == x (the network input). */
    for (size_t li = 0; li < L; ++li) {
        Layer        *l    = &net->layers[li];
        const Matrix *prev = (li == 0) ? x : &net->layers[li - 1].a;
        for (size_t o = 0; o < l->outputs; ++o) {
            double  dl    = l->delta.data[o];
            double *gwrow = l->gW.data + o * l->inputs;
            for (size_t i = 0; i < l->inputs; ++i)
                gwrow[i] = dl * prev->data[i];
            l->gb.data[o] = dl;
        }
    }
    return 0;
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
