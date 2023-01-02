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
BUILD_FLAGS := -O3 -xCore-AVX512 -qopt-zmm-usage=high
ASM_DUMP_FLAGS := -O3 -g3 -xCore-AVX512 -qopt-zmm-usage=high -S

CXX_SOURCES := $(shell find $(SOURCE_DIR) -name '*.cpp')
C_SOURCES := $(shell find $(SOURCE_DIR) -name '*.c')
ASM_DUMP_FILEPATH := ${ASM_DIR}$(filter %.s,$(ASM_DUMP_SOURCE:.cpp=.s))

all: run build asm

run: build
	${BUILD_DIR}${EXE_FILE}

build: ${BUILD_DIR}${EXE_FILE}

asm: ${ASM_DUMP_FILEPATH}


${BUILD_DIR}${EXE_FILE}:
	mkdir -p ${BUILD_DIR}
	./build.sh

${ASM_DUMP_FILEPATH}:
	mkdir -p ${ASM_DIR}
	./asm.sh
	mv *.s ${ASM_DIR}

clean:
	rm -f *.s *.out
	rm -rf ${BUILD_DIR} ${ASM_DIR}

.PHONY: build asm clean run all