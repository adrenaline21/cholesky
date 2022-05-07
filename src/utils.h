#pragma once

#include "sparse.h"

void Visualize(const CSCMatrix& A);
void VisualizeValue(const TCMatrix& A);
void VisualizeValue(const CSCMatrix& A);

void NestedDissection3D(int* a, int l, int m, int n);
void GeneratePoisson3D(CSCMatrix& A, int l, int m, int n);

std::vector<float> RandomVector(int n);
float Error(const std::vector<float>& a, const std::vector<float>& b);
float Norm(const std::vector<float>& a);
