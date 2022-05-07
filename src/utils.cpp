#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <cassert>

void Visualize(const CSCMatrix& A) {
    int n = A.n;
    std::vector<int> v(n * n);
    for (int j = 0; j < n; j++)
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            // v[i * n + j] = std::abs(A.val[p]);
            v[i * n + j] = 1;
        }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (v[i * n + j] != 0) {
                printf("x ");
                // printf("%d ", v[i * n + j]);
            } else {
                printf("  ");
            }
        }
        puts("");
    }
}

void VisualizeValue(const TCMatrix& A) {
    int n = A.n * NB;
    std::vector<float> v(n * n);
    v.clear();
    for (int j = 0; j < A.n; j++)
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            for (int ib = 0; ib < NB; ib++)
                for (int jb = 0; jb < NB; jb++) {
                    v[(i * NB + ib) * n + j * NB + jb] =
                        std::abs(A.val[p * NB * NB + ib * NB + jb]);
                }
        }
    printf("N:%d\n", n);
    float eps = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(v[i * n + j]) > eps) {
                // printf("x ");
                printf("%1.1f ", v[i * n + j]);
            } else {
                printf("    ");
            }
        }
        puts("");
    }
}

void VisualizeValue(const CSCMatrix& A) {
    int n = A.n;
    std::vector<float> v(n * n);
    v.clear();
    for (int j = 0; j < A.n; j++)
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            // printf("%d:%d:%d:%d\n", p, i, j, n);
            v[i * n + j] = std::abs(A.val[p]);
        }
    printf("N:%d\n", n);
    float eps = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(v[i * n + j]) > eps) {
                // printf("x ");
                printf("%1.1f ", v[i * n + j]);
            } else {
                printf("    ");
            }
        }
        puts("");
    }
}

void SubND(int* p, int x0, int x1, int y0, int y1, int z0, int z1, int l, int m,
           int n) {
    int dx = x1 - x0 + 1, dy = y1 - y0 + 1, dz = z1 - z0 + 1;
    if (std::max({dx, dy, dz}) <= 2) {
        for (int i = x0; i <= x1; i++)
            for (int j = y0; j <= y1; j++)
                for (int k = z0; k <= z1; k++) {
                    // printf("%d %d %d\n", i, j, k);
                    p[(i - x0) * dy * dz + (j - y0) * dz + (k - z0)] =
                        i * m * n + j * n + k;
                }
        return;
    }
    if (dx >= dy && dx >= dz) {
        int mx = (x0 + x1) / 2;
        int offset1 = (mx - x0) * dy * dz,
            offset2 = offset1 + (x1 - mx) * dy * dz;
        SubND(p, x0, mx - 1, y0, y1, z0, z1, l, m, n);
        SubND(p + offset1, mx + 1, x1, y0, y1, z0, z1, l, m, n);
        SubND(p + offset2, mx, mx, y0, y1, z0, z1, l, m, n);
    } else if (dy >= dx && dy >= dz) {
        int my = (y0 + y1) / 2;
        int offset1 = (my - y0) * dx * dz,
            offset2 = offset1 + (y1 - my) * dx * dz;
        SubND(p, x0, x1, y0, my - 1, z0, z1, l, m, n);
        SubND(p + offset1, x0, x1, my + 1, y1, z0, z1, l, m, n);
        SubND(p + offset2, x0, x1, my, my, z0, z1, l, m, n);
    } else {
        int mz = (z0 + z1) / 2;
        int offset1 = (mz - z0) * dx * dy,
            offset2 = offset1 + (z1 - mz) * dx * dy;
        SubND(p, x0, x1, y0, y1, z0, mz - 1, l, m, n);
        SubND(p + offset1, x0, x1, y0, y1, mz + 1, z1, l, m, n);
        SubND(p + offset2, x0, x1, y0, y1, mz, mz, l, m, n);
    }
}

void NestedDissection3D(int* p, int l, int m, int n) {
    SubND(p, 0, l - 1, 0, m - 1, 0, n - 1, l, m, n);
}

// In place (?) factorization, so storing both L and U.
void GeneratePoisson3D(CSCMatrix& A, int l, int m, int n) {
    int N = l * m * n;
    A.n = N;
    A.out.resize(N + 1);
    A.in.clear();
    A.val.clear();
    for (int i = 0; i < l; i++)
        for (int j = 0; j < m; j++)
            for (int k = 0; k < n; k++) {
                // printf("%d %d %d %d\n", i, j, k, sum);
                int idx = i * m * n + j * n + k;
                A.out[idx] = A.val.size();
                int vdiag = 1;
                if (i > 0) {
                    A.in.push_back(idx - m * n);
                    A.val.push_back(-1);
                    vdiag++;
                }
                if (j > 0) {
                    A.in.push_back(idx - n);
                    A.val.push_back(-1);
                    vdiag++;
                }
                if (k > 0) {
                    A.in.push_back(idx - 1);
                    A.val.push_back(-1);
                    vdiag++;
                }
                if (k < n - 1) {
                    A.in.push_back(idx + 1);
                    A.val.push_back(-1);
                    vdiag++;
                }
                if (j < m - 1) {
                    A.in.push_back(idx + n);
                    A.val.push_back(-1);
                    vdiag++;
                }
                if (i < l - 1) {
                    A.in.push_back(idx + m * n);
                    A.val.push_back(-1);
                }
                A.in.push_back(idx);
                A.val.push_back(vdiag);
            }
    A.out[N] = A.val.size();
    // for (int i = 0; i <= N; i++) printf("%d ", A.out[i]);
    // puts("");
}

std::vector<float> RandomVector(int n) {
    std::vector<float> x(n);
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dis(-1., 1.);
    for (int i = 0; i < n; i++) x[i] = dis(gen);
    return x;
}

float Error(const std::vector<float>& a, const std::vector<float>& b) {
    int n = a.size();
    assert(a.size() == b.size());
    float sum = 0;
    for (int i = 0; i < n; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(sum);
}

float Norm(const std::vector<float>& a) {
    int n = a.size();
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * a[i];
    return sqrt(sum);
}