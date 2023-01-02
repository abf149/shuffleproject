# Experiment: in-register 16x16 matrix transpose

You must install [Intel® oneAPI DPC++/C++ Compiler (icpx) or Intel C++ Compiler Classic (icpc)](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) and *make* to compile this project.

**This will only run on a machine with AVX512 support.**

**Description:** build a test executable to demonstrate in-place in-register SIMD 16x16 matrix transpose using the O(nlogn) tran_new2 algorithm [here.](https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions)

Build & run:
```
make all # build, run test, and dump optimized assembly code for transpose
make run # build, run test
make build # build test executable
make asm # dump optimized assembly code for transpose
```

Use `make clean` to clear compiler outputs.

## Contents

* **src/naive_xpose.cpp** - a completely naive O(n^2) transpose implementation.
* **src/intrinsic_transpose.cpp** - O(nlogn) in-register SIMD 16x16 matrix transpose based on the tran_new2 function developed [in this StackOverflow post.](https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions)
    * **intrinsic_transpose.s.bk** - assembly dump of in-register SIMD 16x16 matrix transpose compiled with -O3 and -g3. The purpose is to support developing a transpose JIT.
* Not finished: XBYAK JIT for in-register SIMD 16x16 matrix transpose based on the assembly dump.