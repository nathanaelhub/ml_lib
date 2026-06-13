/* ------------------------------------------------------------------ *
 *  export_viz.c - dump data for the project's visualisations.
 *
 *  This links against the real ml_lib (it is itself a small client of
 *  the library), trains the same models the demos use, and writes a few
 *  CSVs that tools/make_plots.py turns into PNGs. The plots therefore
 *  reflect what the C library actually computes.
 *
 *  Build & run from the repo root, e.g. with MSVC:
 *      cl /W4 /O2 ml_lib.c tools\export_viz.c /Fe:export_viz.exe
 *      export_viz.exe
 *  or gcc:
 *      gcc -O2 ml_lib.c tools/export_viz.c -lm -o export_viz && ./export_viz
 * ------------------------------------------------------------------ */
#include "ml_lib.h"

#include <stdio.h>
#include <stdlib.h>

/* Train a 2-8-1 net on the XOR-of-signs data and dump per-epoch loss. */
static void dump_loss(const char *path, const Dataset *d, int use_adam,
                      int epochs, size_t batch, double lr)
{
    const size_t     sizes[] = { 2, 8, 1 };
    const Activation acts[]  = { ACT_TANH, ACT_SIGMOID };
    Network   net = net_alloc(sizes, acts, 2);
    Optimizer opt = adam_default(lr);

    FILE *f = fopen(path, "w");
    if (f == NULL) { net_free(&net); return; }
    fprintf(f, "epoch,loss\n");
    for (int e = 0; e < epochs; ++e) {
        double loss = use_adam ? net_train_epoch_adam(&net, d, batch, &opt)
                               : net_train_epoch(&net, d, batch, lr);
        fprintf(f, "%d,%.8f\n", e, loss);
    }
    fclose(f);
    net_free(&net);
}

/* Train well, then evaluate a dense grid so we can draw the learned
 * decision surface (probability of class 1 at each point). */
static void dump_decision_grid(const char *path, const Dataset *d)
{
    const size_t     sizes[] = { 2, 8, 1 };
    const Activation acts[]  = { ACT_TANH, ACT_SIGMOID };
    Network   net = net_alloc(sizes, acts, 2);
    Optimizer opt = adam_default(0.01);
    for (int e = 0; e < 300; ++e)
        net_train_epoch_adam(&net, d, 16, &opt);

    FILE *f = fopen(path, "w");
    if (f == NULL) { net_free(&net); return; }
    fprintf(f, "x1,x2,prob\n");

    const int    N  = 140;
    const double lo = -1.15, hi = 1.15;
    Matrix x = mat_alloc(2, 1);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x1 = lo + (hi - lo) * i / (N - 1);
            double x2 = lo + (hi - lo) * j / (N - 1);
            x.data[0] = x1;
            x.data[1] = x2;
            double p = net_forward(&net, &x)->data[0];
            fprintf(f, "%.5f,%.5f,%.5f\n", x1, x2, p);
        }
    }
    mat_free(&x);
    fclose(f);
    net_free(&net);
}

/* Train the Iris softmax net on 80% and dump the 3x3 confusion matrix
 * over the held-out 20% validation set. */
static void dump_iris_confusion(const char *path)
{
    Dataset d = dataset_load_csv("data/iris.csv", 4, 3, 1);
    if (d.X.data == NULL)
        return;

    Dataset tr, va;
    if (dataset_split(&d, 0.2, &tr, &va) != 0) { dataset_free(&d); return; }

    const size_t     sizes[] = { 4, 16, 3 };
    const Activation acts[]  = { ACT_TANH, ACT_SOFTMAX };
    Network   net = net_alloc(sizes, acts, 2);
    Optimizer opt = adam_default(0.01);
    for (int e = 0; e < 120; ++e)
        net_train_epoch_adam(&net, &tr, 16, &opt);

    int conf[3][3] = { { 0 } };
    for (size_t i = 0; i < va.n_samples; ++i) {
        Matrix xv    = dataset_input(&va, i);
        Matrix tv    = dataset_target(&va, i);
        size_t pred  = mat_argmax(net_forward(&net, &xv));
        size_t truth = mat_argmax(&tv);
        conf[truth][pred] += 1;
    }

    FILE *f = fopen(path, "w");
    if (f != NULL) {
        fprintf(f, "true,pred,count\n");
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                fprintf(f, "%d,%d,%d\n", a, b, conf[a][b]);
        fclose(f);
    }

    net_free(&net);
    dataset_free(&tr);
    dataset_free(&va);
    dataset_free(&d);
}

int main(void)
{
    srand(12345u);    /* match the demos' deterministic seed */

    Dataset data_xor = dataset_load_csv("data/xor2d.csv", 2, 1, 1);
    if (data_xor.X.data == NULL) {
        fprintf(stderr, "export_viz: run from the repo root (needs data/xor2d.csv)\n");
        return 1;
    }

    dump_loss("viz_loss_sgd.csv",  &data_xor, 0, 150, 16, 0.5);
    dump_loss("viz_loss_adam.csv", &data_xor, 1, 150, 16, 0.01);
    dump_decision_grid("viz_grid.csv", &data_xor);
    dataset_free(&data_xor);

    dump_iris_confusion("viz_iris_confusion.csv");

    printf("export_viz: wrote viz_loss_sgd.csv, viz_loss_adam.csv, "
           "viz_grid.csv, viz_iris_confusion.csv\n");
    return 0;
}
