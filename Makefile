# Make transpose test executable
#
# Options:
# make build - compile test executable
# make run - compile and run test executable
# make asm - dump optimized assembly code for SIMD in-register 16x16 transpose
# make all - all of the above
# make clean

.EXPORT_ALL_VARIABLES:

INTEL_PATH := /opt/intel/oneapi/
CXX := icpc
INCLUDE_PATH_FLAGS := -IHowToOptimizeGemm/ -Iinc/
SOURCE_DIR := src/
FLAME_GEMM_SOURCE_FILES := HowToOptimizeGemm/random_matrix.c HowToOptimizeGemm/print_matrix.c
ASM_DUMP_SOURCE := intrinsic_transpose.cpp
BUILD_DIR := build/
EXE_FILE := test_transpose.out
ASM_DIR := asm/
CXX_FLAGS := -O3 -xCore-AVX512 -qopt-zmm-usage=high -c
C_FLAGS := -O3 -xCore-AVX512 -qopt-zmm-usage=high -c
JIT_CXX_FLAGS := -O3 -c
LINK_FLAGS := -O3 -L
ASM_DUMP_FLAGS := -O3 -g3 -xCore-AVX512 -qopt-zmm-usage=high -S

CXX_SOURCES := $(filter-out $(SOURCE_DIR)/jit_transpose.cpp,$(shell find $(SOURCE_DIR) -name '*.cpp'))
C_SOURCES := $(shell find $(SOURCE_DIR) -name '*.c')
ASM_DUMP_FILEPATH := ${ASM_DIR}$(filter %.s,$(ASM_DUMP_SOURCE:.cpp=.s))

all: run build asm

run: build
	${BUILD_DIR}${EXE_FILE}

build: ${BUILD_DIR}${EXE_FILE}

asm: ${ASM_DUMP_FILEPATH}

#$(CXX_SOURCES:.cpp=.o):
#	./build_cxx.sh

#$(C_SOURCES:.cpp=.o):
#	./build_c.sh

*.o:
	./build_cxx.sh
	./build_c.sh

#jit_transpose.o:
#	g++ -O3 src/jit_transpose.cpp

${BUILD_DIR}${EXE_FILE}: *.o
	mkdir -p ${BUILD_DIR}
	gcc $(LINK_FLAGS) -o  $@ $^

${ASM_DUMP_FILEPATH}:
	mkdir -p ${ASM_DIR}
	./asm.sh
	mv *.s ${ASM_DIR}

clean:
	rm -f *.s *.out
	rm -rf ${BUILD_DIR} ${ASM_DIR}

.PHONY: build asm clean run all