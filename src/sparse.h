#pragma once

#include <vector>
#include <map>

#define NB 16

struct CSCMatrix {
    CSCMatrix(){};
    CSCMatrix(int n) {
        this->n = n;
        (this->in).resize(n);
        (this->etree).resize(n);
    };
    int n;
    std::vector<int> in;     // row indicies, nnz
    std::vector<int> out;    // col pre_sum, n+1
    std::vector<float> val;  // nnz
    std::vector<int> etree;
};

struct TCMatrix : CSCMatrix {
    std::map<int, int> offsets;  // <col-maj block idx, offset in val>
};

void Permute(const CSCMatrix& A, int* perm, CSCMatrix& F);

void ETree(CSCMatrix& A);

void FillIn(CSCMatrix& A, CSCMatrix& L);

void Compress(CSCMatrix& A, TCMatrix& L);

void FactorizeTC(TCMatrix& L);

void FactorizeNaive(CSCMatrix& A, CSCMatrix& L);

// std::vector<float> Multiply(const CSCMatrix& A, const std::vector<float>& b);
std::vector<float> Potrs(const CSCMatrix& L, std::vector<float>& b);
std::vector<float> Potrs(const TCMatrix& L, std::vector<float>& b);

void Gemv(const TCMatrix& A, std::vector<float>& x, std::vector<float>& b,
          float alpha = 1.0, float beta = 1.0);
void Gemv(const CSCMatrix& A, std::vector<float>& x, std::vector<float>& b,
          float alpha = 1.0, float beta = 1.0);