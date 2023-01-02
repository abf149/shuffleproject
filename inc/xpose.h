#ifndef H_XPOSE_H
#define H_XPOSE_H

typedef int DATA_TYPE;

void naive_square_transpose(int m, int k, DATA_TYPE* a, int lda);

void simd_square_transpose_intrinsic_16x16(int m, int k, DATA_TYPE* a, int lda);

void simd_tran_16x16_permutexvar_inplace(int* mat);

void simd_tran_16x16_permutexvar_outofplace(int* mat, int* matT);

void tran_new2(int* mat, int* matT);

#endif