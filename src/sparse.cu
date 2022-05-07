#include "sparse.h"
#include "utils.h"

#include <cstdio>
#include <set>
#include <algorithm>
#include <cassert>

#include <cublas_v2.h>
#include <mma.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define EMPTY (-1)

void Permute(const CSCMatrix& A, int* perm, CSCMatrix& F) {
    int n = A.n, nnz = A.out[n];
    // printf("%d\n", nnz);
    F.n = n;
    F.out.resize(n + 1);
    F.in.resize(nnz);
    F.val.resize(nnz);
    int pinv[n];
    for (int i = 0; i < n; i++) {
        pinv[perm[i]] = i;
    }

    int nz[n];
    for (int i = 0; i < n; i++) nz[i] = 0;
    // Loop over A to find F.out
    int i_old, j_old, i;
    for (int j = 0; j < n; j++) {
        j_old = perm[j];
        for (int p = A.out[j_old]; p < A.out[j_old + 1]; p++) {
            i_old = A.in[p];
            // printf("%d %d\n", i_old, j_old);
            i = pinv[i_old];
            nz[j]++;
        }
    }

    int sum = 0;
    for (int i = 0; i <= n; i++) {
        F.out[i] = sum;
        sum += nz[i];
    }
    // for (int i = 0; i <= n; i++) printf("%d ", F.out[i]);
    // puts("");

    // Loop again to fill in F
    int idx;
    for (int j = 0; j < n; j++) {
        j_old = perm[j];
        for (int p = A.out[j_old]; p < A.out[j_old + 1]; p++) {
            i_old = A.in[p];
            // printf("%d %d\n", i_old, j_old);
            i = pinv[i_old];

            idx = F.out[j]++;
            F.in[idx] = i;
            F.val[idx] = A.val[p];
        }
    }
    for (int i = n - 1; i > 0; i--) F.out[i] = F.out[i - 1];
    F.out[0] = 0;

    // for (int i = 0; i <= n; i++) printf("%d ", F.out[i]);
    // puts("");
    // for (int i = 0; i <= nnz; i++) printf("%d ", F.in[i]);
    // puts("");
    // for (int i = 0; i <= nnz; i++) printf("%d ", F.val[i]);
    // puts("");
}

void updateETree(int k, int i, int* parent, int* ancestor) {
    int a;
    while (true) {
        a = ancestor[k];
        if (a == i) return;
        ancestor[k] = i;
        if (a == EMPTY) {
            parent[k] = i;
            return;
        }
        k = a;
    }
}

void ETree(CSCMatrix& A) {
    int n = A.n;
    A.etree.resize(n);
    int* parent = A.etree.data();
    int ancestor[n];
    for (int i = 0; i < n; i++) {
        ancestor[i] = EMPTY;
        parent[i] = EMPTY;
    }

    for (int j = 0; j < n; j++) {
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            if (i < j) updateETree(i, j, parent, ancestor);
        }
    }

    // for (int i = 0; i < n; i++) printf("%d %d\n", i, parent[i]);
}

// Symbolic fill LL'=A, symmetric A fully stored.
void FillIn(CSCMatrix& A, CSCMatrix& L) {
    ETree(A);
    int n = A.n;
    L.n = n;
    L.out.resize(n + 1);
    L.in.clear();
    bool P[n];
    for (int i = 0; i < n; i++) P[i] = false;
    for (int j = 0; j < n; j++) {
        L.out[j] = L.in.size();
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            if (i >= j) {
                P[i] = true;
                L.in.push_back(i);
            }
        }
        for (int js = 0; js < j; js++)
            if (A.etree[js] == j) {
                for (int p = L.out[js]; p < L.out[js + 1]; p++) {
                    int i = L.in[p];
                    if (P[i] == false && i > j) {
                        P[i] = true;
                        L.in.push_back(i);
                    }
                }
            }
        for (int p = L.out[j]; p < L.in.size(); p++) P[L.in[p]] = false;
    }
    L.out[n] = L.in.size();
    L.val.resize(L.out[n]);
}

// Compress into 16*16 blocks. Each block row-major for cublas trsm
void Compress(CSCMatrix& A, TCMatrix& L) {
    std::set<int> iR;
    int n = A.n;
    int N = (n + NB - 1) / NB;
    CSCMatrix R(n);
    R.n = N;
    R.out.resize(N + 1);
    R.in.clear();

    int j, p, i, idx, off;
    for (j = 0; j < n; j++)
        for (p = A.out[j]; p < A.out[j + 1]; p++) {
            i = A.in[p];
            iR.insert((i / NB) * N + j / NB);
        }
    int nz[N];
    for (i = 0; i < N; i++) nz[i] = 0;
    for (auto id : iR) nz[id % N]++;
    int sum = 0;
    for (j = 0; j <= N; j++) {
        R.out[j] = sum;
        sum += nz[j];
    }
    // for (i = 0; i <= N; i++) printf("%d ", R.out[i]);
    // puts("");
    int nnz = R.out[N];
    R.in.resize(nnz);
    for (auto id : iR) {
        j = id % N, i = id / N;
        idx = R.out[j]++;
        R.in[idx] = i;
    }
    for (i = N - 1; i > 0; i--) R.out[i] = R.out[i - 1];
    R.out[0] = 0;
    // printf("%d \n", N);

    ETree(R);
    FillIn(R, L);
    nnz = L.in.size();
    // build mappings
    L.offsets.clear();
    using Pair = std::pair<int, int>;
    int* Li = L.in.data();
    for (j = 0; j < N; j++) {
        // sort the row-idx so that the diagonal blocks are always at the
        // beginning of the memory
        std::sort(Li + L.out[j], Li + L.out[j + 1]);
        for (p = L.out[j]; p < L.out[j + 1]; p++) {
            i = L.in[p];
            idx = i + j * N;
            L.offsets.insert(Pair(idx, p));
        }
    }
    // for (auto p : L.offsets) printf("%d %d\n", p.first, p.second);

    // fill in values
    L.val.resize(nnz * NB * NB);
    for (j = 0; j < n; j++)
        for (p = A.out[j]; p < A.out[j + 1]; p++) {
            i = A.in[p];
            if (i / NB >= j / NB) {
                idx = L.offsets.find((i / NB) + (j / NB) * N)->second;
                // printf("%d:%d:%d\n", i, j, idx);
                off = (i % NB) * NB + (j % NB);
                L.val[idx * NB * NB + off] = A.val[p];
            }
        }
}

// Import from LAPACK
extern "C" {
extern int dpotrf_(char*, int*, float*, int*, int*);
extern int spotrf_(char*, int*, float*, int*, int*);
}

#define cudaErrCheck(stat) \
    { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char* file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat),
                file, line);
    }
}

#define cublasErrCheck(stat) \
    { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char* file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

__global__ void convertFp32ToFp16(half* out, float* in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void convertFp16ToFp32(float* out, half* in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void wmma_ker(half* a, float* L, int* i, int* j, int* off) {
    int x = blockIdx.x;
    // printf("%d %d %d %d\n", x, i[x], j[x], off[x]);
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a + NB * NB * j[x], NB);
    wmma::load_matrix_sync(b_frag, a + NB * NB * i[x], NB);
    wmma::fill_fragment(acc_frag, 0.0f);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::load_matrix_sync(c_frag, L + NB * NB * off[x], NB,
                           wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] -= acc_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(L + NB * NB * off[x], c_frag, 16,
                            wmma::mem_row_major);
}

// 16*16 blocked right-looking cholesky LL'
void FactorizeTC(TCMatrix& L) {
    // Setup cublas for trsm.
    const float d_one = 1.0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
    size_t block_size = NB * NB * sizeof(float);

    // LAPACK setup. Could be using cuSolver to reduce memory transfer, but too
    // complex to set up. Could also set up a potrf16*16 kernel.
    int n = L.n, nb = NB, info;
    char order = 'U';
    float* v = L.val.data();
    float W[NB * NB];

    // Move all data to device
    float* dL;
    int nnz = L.out[n];
    printf("\#TCMatrix blocks:%d\n", nnz);
    cudaMalloc((void**)&dL, nnz * block_size);
    cudaMemcpy(dL, v, nnz * block_size, cudaMemcpyHostToDevice);

    int idx;
    for (int k = 0; k < n; k++) {
        // idx = L.offsets.find(k * n + k)->second;
        // sorted, so
        idx = L.out[k];
        cudaMemcpy(W, dL + idx * NB * NB, block_size, cudaMemcpyDeviceToHost);
        spotrf_(&order, &nb, W, &nb, &info);
        if (info != 0) printf("Diagonal Factorization Failed!");
        // for (int i = 0; i < 256; i++) printf("%f ", W[i]);
        cudaMemcpy(dL + idx * NB * NB, W, block_size, cudaMemcpyHostToDevice);
        // for (int i = 0; i < 256; i++) W[i] = 0;
        // cudaMemcpy(W, dL + idx * NB * NB, block_size,
        // cudaMemcpyDeviceToHost); for (int i = 0; i < 256; i++) printf("%f ",
        // W[i]);
        int M = L.out[k + 1] - L.out[k] - 1;
        size_t a_size = M * block_size;
        cublasErrCheck(
            cublasStrsm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, NB, M * NB, &d_one,
                        dL + idx * NB * NB, NB, dL + (idx + 1) * NB * NB, NB));

        int a_num = M * NB * NB;
        half* dAh;
        cudaMalloc((void**)&dAh, a_num * sizeof(half));
        convertFp32ToFp16<<<(a_num + 255) / 256, 256>>>(
            dAh, dL + (idx + 1) * NB * NB, a_num);

        // float* dA;
        // float W1[a_num], W2[a_num];
        // cudaMalloc((void**)&dA, a_num * sizeof(float));
        // convertFp16ToFp32<<<(a_num + 255) / 256, 256>>>(dA, dAh, a_num);
        // cudaMemcpy(W1, dA, a_num * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(W2, dL + (idx + 1) * NB * NB, a_num * sizeof(float),
        //            cudaMemcpyDeviceToHost);
        // for (int i = 0; i < a_num; i++) printf("%f %f\n", W1[i], W2[i]);

        // gemm part, build offsets
        int nPair = M * (M + 1) / 2;
        int ii[nPair], jj[nPair], offsets[nPair], i = 0;
        for (int p1 = L.out[k] + 1; p1 < L.out[k + 1]; p1++) {
            for (int p2 = p1; p2 < L.out[k + 1]; p2++) {
                // printf("%d:%d:%d:%d\n", p1, L.in[p1], p2, L.in[p2]);
                assert(L.in[p1] <= L.in[p2]);  // sorted
                ii[i] = p1 - L.out[k] - 1;
                jj[i] = p2 - L.out[k] - 1;
                auto ite = L.offsets.find(L.in[p2] + L.in[p1] * n);
                assert(ite != L.offsets.end());
                offsets[i] = ite->second;
                // printf("%d:%d:%d\n", ii[i], jj[i], offsets[i]);
                i++;
            }
        }

        int *dI, *dJ, *dO;
        cudaErrCheck(cudaMalloc((void**)&dI, nPair * sizeof(int)));
        cudaErrCheck(
            cudaMemcpy(dI, ii, nPair * sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void**)&dJ, nPair * sizeof(int)));
        cudaErrCheck(
            cudaMemcpy(dJ, jj, nPair * sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void**)&dO, nPair * sizeof(int)));
        cudaErrCheck(cudaMemcpy(dO, offsets, nPair * sizeof(int),
                                cudaMemcpyHostToDevice));

        wmma_ker<<<nPair, 32>>>(dAh, dL, dI, dJ, dO);

        cudaFree(dO);
        cudaFree(dJ);
        cudaFree(dI);
        cudaFree(dAh);
    }
    cudaMemcpy(v, dL, nnz * block_size, cudaMemcpyDeviceToHost);

    // CUDA cleanup.
    cudaFree(dL);
    cudaDeviceReset();
}

// The most dumb right-looking cholesky, and not in-place
void FactorizeNaive(CSCMatrix& A, CSCMatrix& L) {
    FillIn(A, L);
    int n = L.n;
    int* Li = L.in.data();
    for (int j = 0; j < n; j++) {
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            std::sort(Li + L.out[j], Li + L.out[j + 1]);
            if (i >= j) {
                for (int k = L.out[j]; k < L.out[j + 1]; k++) {
                    if (L.in[k] == i) {
                        L.val[k] = A.val[p];
                        break;
                    }
                }
            }
        }
    }

    for (int j = 0; j < n; j++) {
        int p = L.out[j];
        assert(L.in[p] == j);
        float l = 1. / sqrt(L.val[p]);
        for (; p < L.out[j + 1]; p++) L.val[p] *= l;
        for (int p1 = L.out[j] + 1; p1 < L.out[j + 1]; p1++) {
            int i = L.in[p1];
            for (int p2 = p1; p2 < L.out[j + 1]; p2++) {
                for (int k = L.out[i]; k < L.out[i + 1]; k++) {
                    if (L.in[k] == L.in[p2]) {
                        L.val[k] -= L.val[p1] * L.val[p2];
                        break;
                    }
                }
            }
        }
    }
}

//
std::vector<float> Potrs(const CSCMatrix& L, std::vector<float>& b) {
    int n = L.n;
    assert(b.size() == n);
    // Ly = b
    std::vector<float> x = b;
    for (int j = 0; j < n; j++) {
        int p = L.out[j];
        assert(L.in[p] == j);
        x[j] /= L.val[p];
        p++;
        for (; p < L.out[j + 1]; p++) {
            int i = L.in[p];
            assert(i > j);
            x[i] -= L.val[p] * x[j];
        }
    }
    // L'x = y
    for (int j = n - 1; j >= 0; j--) {
        int p = L.out[j];
        assert(L.in[p] == j);
        p++;
        for (; p < L.out[j + 1]; p++) {
            int i = L.in[p];
            assert(i > j);
            x[j] -= L.val[p] * x[i];
        }
        x[j] /= L.val[L.out[j]];
    }

    return x;
}

std::vector<float> Potrs(const TCMatrix& L, std::vector<float>& b) {
    int n = L.n;
    assert(b.size() == n * NB);
    // Ly = b
    std::vector<float> x = b;
    for (int j = 0; j < n; j++) {
        int p = L.out[j];
        assert(L.in[p] == j);
        // Row-major diagonal elimination
        for (int ii = 0; ii < NB; ii++) {
            int row_start = p * NB * NB + ii * NB;
            for (int jj = 0; jj < ii; jj++) {
                x[j * NB + ii] -= L.val[row_start + jj] * x[j * NB + jj];
            }
            x[j * NB + ii] /= L.val[row_start + ii];
        }
        p++;
        for (; p < L.out[j + 1]; p++) {
            int i = L.in[p];
            assert(i > j);
            // x[i] -= L.val[p] * x[j];
            for (int ii = 0; ii < NB; ii++) {
                int row_start = p * NB * NB + ii * NB;
                for (int jj = 0; jj < NB; jj++) {
                    x[i * NB + ii] -= L.val[row_start + jj] * x[j * NB + jj];
                }
            }
        }
    }
    // L'x = y
    for (int j = n - 1; j >= 0; j--) {
        int p = L.out[j];
        assert(L.in[p] == j);
        p++;
        for (; p < L.out[j + 1]; p++) {
            int i = L.in[p];
            assert(i > j);
            // x[j] -= L.val[p] * x[i]; Transposed.
            for (int jj = 0; jj < NB; jj++) {
                int col_start = p * NB * NB + jj;
                for (int ii = 0; ii < NB; ii++) {
                    x[j * NB + jj] -=
                        L.val[col_start + ii * NB] * x[i * NB + ii];
                }
            }
        }
        // L' elimination, transposed, so col_start really row_start in L
        p = L.out[j];
        for (int jj = NB - 1; jj >= 0; jj--) {
            int col_start = p * NB * NB + jj * NB;
            x[j * NB + jj] /= L.val[col_start + jj];
            for (int ii = 0; ii < jj; ii++) {
                x[j * NB + ii] -= L.val[col_start + ii] * x[j * NB + jj];
            }
        }
    }

    return x;
}

// b = alpha Ax + beta b.
// NOTE: TCMatrix is Lower, so diagonal not fully used.
void Gemv(const TCMatrix& A, std::vector<float>& x, std::vector<float>& b,
          float alpha, float beta) {
    int n = A.n;
    for (int i = 0; i < n; i++) b[i] *= beta;
    for (int j = 0; j < n; j++) {
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            if (i == j) {
                for (int ii = 0; ii < NB; ii++) {
                    int row_start = p * NB * NB + ii * NB;
                    for (int jj = 0; jj <= ii; jj++) {
                        b[i * NB + ii] +=
                            alpha * A.val[row_start + jj] * x[j * NB + jj];
                    }
                }
            } else {
                for (int ii = 0; ii < NB; ii++) {
                    int row_start = p * NB * NB + ii * NB;
                    for (int jj = 0; jj < NB; jj++) {
                        b[i * NB + ii] +=
                            alpha * A.val[row_start + jj] * x[j * NB + jj];
                    }
                }
            }
        }
    }
}

void Gemv(const CSCMatrix& A, std::vector<float>& x, std::vector<float>& b,
          float alpha, float beta) {
    int n = A.n;
    assert(n == b.size());
    for (int i = 0; i < n; i++) b[i] *= beta;
    for (int j = 0; j < n; j++) {
        for (int p = A.out[j]; p < A.out[j + 1]; p++) {
            int i = A.in[p];
            b[i] += alpha * A.val[p] * x[j];
        }
    }
}

// WIP
// void FactorizeMultifrontal(CSCMatrix& A, CSCMatrix& L) {
//     int n = A.n, k;
//     int Lnz[n];
//     bool flag[n];
//     for (k = 0; k < n; k++) {
//         Lnz[k] = 1;
//         flag[k] = false;
//     }
//     int top, i, len;
//     int stack[n];
//     float W[n];
//     for (k = 0; k < n; k++) {
//         top = n;
//         flag[k] = true;
//         for (int p = A.out[k]; p < A.out[k + 1]; p++) {
//             i = A.in[p];
//             if (i <= k) {
//                 W[i] = A.val[p];
//                 for (len = 0; i < k && !flag[i] && i != EMPTY; i =
//                 A.etree[i]) {
//                     stack[len++] = i;
//                     flag[i] = true;
//                 }
//                 while (len > 0) stack[--top] = stack[--len];
//             }
//         }
//     }
// }
