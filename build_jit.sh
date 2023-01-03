source ${INTEL_PATH}setvars.sh
${CXX} ${JIT_CXX_FLAGS} ${INCLUDE_PATH_FLAGS} src/jit_transpose.cpp

echo ${CXX_SOURCES}

# rm test_xpose.out

# if command -v icpx &> /dev/null
# then

# source /home/tamdev/intel/oneapi/setvars.sh
# icpx -O3 -xCore-AVX512 -qopt-zmm-usage=high -IHowToOptimizeGemm/ HowToOptimizeGemm/random_matrix.c HowToOptimizeGemm/print_matrix.c intrinsic_transpose.cpp main.cpp naive_xpose.cpp test_xpose.cpp -o test_xpose.out

# ./test_xpose.out

# fi

# if command -v icpc &> /dev/null
# then

# source /opt/intel/oneapi/setvars.sh
# icpc -O3 -g3 -xCore-AVX512 -qopt-zmm-usage=high -IHowToOptimizeGemm/ HowToOptimizeGemm/random_matrix.c HowToOptimizeGemm/print_matrix.c intrinsic_transpose.cpp main.cpp naive_xpose.cpp test_xpose.cpp -S

# fi
