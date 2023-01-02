# Experiment: in-register 16x16 matrix transpose

You must install [Intel® oneAPI DPC++/C++ Compiler (icpx) or Intel C++ Compiler Classic (icpc)](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) and *make* to compile this project:


```
# Compile

```

## Contents

* **src/naive_xpose.cpp** - a completely naive O(n^2) transpose implementation.
* **src/intrinsic_transpose.cpp** - in-register SIMD 16x16 matrix transpose based on the tran_new2 function developed [in this StackOverflow post.](https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions)
    * **intrinsic_transpose.s.bk** - assembly dump of in-register SIMD 16x16 matrix transpose compiled with -O3 and -g3. The purpose is to support developing a transpose JIT.
* Not finished: XBYAK JIT for in-register SIMD 16x16 matrix transpose based on the assembly dump.