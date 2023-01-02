#include <iostream>
#include "test_transpose.h"

int main() {
    //test_print_naive_square_transpose();
    test_print_simd_square_transpose_intrinsic_16x16();
}