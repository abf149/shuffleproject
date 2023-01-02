#ifndef H_TEST_XPOSE_H
#define H_TEST_XPOSE_H

/* University of Texas flame gemm support functions */
void REF_MMult(int, int, int, double *, int, double *, int, double *, int );
void MY_MMult(int, int, int, double *, int, double *, int, double *, int );
void copy_matrix(int, int, double *, int, double *, int );
void random_matrix(int, int, double *, int);
double compare_matrices( int, int, double *, int, double *, int );
void print_matrix( int m, int n, double *a, int lda );
void print_matrix( int m, int n, int *a, int lda );
void dummy_int_matrix( int m, int n, int *a, int lda );
double dclock();

/* Regressions */
void test_print_naive_square_transpose();
void test_print_simd_square_transpose_intrinsic_16x16();

#endif