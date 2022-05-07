#include "sparse.h"
#include "utils.h"

#include <vector>
#include <cstdio>
#include <algorithm>

// To align with 16*16 block, should be multiple of 4.
const int N = 32;

int main() {
    std::vector<int> p(N * N * N);
    NestedDissection3D(p.data(), N, N, N);
    // for (int i = 0; i < N * N * N; ++i) printf("%d ", p[i]);
    // puts("");

    int n = N * N * N;
    CSCMatrix A(n), F(n), L(n);
    GeneratePoisson3D(A, N, N, N);
    // Visualize(A);

    FillIn(A, L);
    // Visualize(L);
    printf("NNZ without nested dissection:%ld\n", L.in.size());

    Permute(A, p.data(), F);
    // Visualize(F);
    FillIn(F, L);
    printf("NNZ after nested dissection:%ld\n", L.in.size());

    auto x0 = RandomVector(n);
    std::vector<float> b(n), x(n);
    Gemv(F, x0, b);

    // puts("b:");
    // for (auto f : b) printf("%1.2f ", f);
    // puts("");

    // puts("x0:");
    // for (auto f : x0) printf("%1.2f ", f);
    // puts("");

    // puts("x:");
    // for (auto f : x) printf("%1.2f ", f);
    // puts("");

    // FactorizeNaive(F, L);
    // // VisualizeValue(L);
    // // Visualize(L);
    // x = Potrs(L, b);
    // printf("Absolute L2 Error:%f\n", Error(x, x0));

    TCMatrix Lf;
    Compress(F, Lf);
    // VisualizeValue(Lf);
    FactorizeTC(Lf);
    // VisualizeValue(Lf);

    // IR:
    float err, tol = 1e-9;
    auto r = b;

    float norm = Norm(x0);
    for (;;) {
        auto xnew = Potrs(Lf, r);
        for (int i = 0; i < n; i++) {
            x[i] += xnew[i];
            // printf("%f ", r[i]);
        }
        err = Error(x, x0);
        float rnorm = Norm(r);
        if (rnorm < tol) break;
        printf("Relative L2 Error:%.8f, Residual Norm:%.8f\n", err / norm,
               rnorm);
        Gemv(F, xnew, r, -1.0, 1.0);
    }

    return 0;
}