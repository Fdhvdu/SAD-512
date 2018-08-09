# SAD-512
Calculate Sum of Absolute Difference (SAD) by AVX-512.<br>
Your machine should support C++17, AVX2 and AVX-512.

# Warnings
Before using this library, you should aware of something.<br>
1.<br>
The code is optimized by the following environments.<br>
```
Motherboard: ASUS TUF X299 MARK 1
CPU: Intel(R) Core(TM) i7-7820X
Memory: 16 GiB DDR4 2400 MHz * 2
OS: Linux 4.17.8-1-ARCH #1 SMP PREEMPT
Compiler information:
	clang version 6.0.1 (tags/RELEASE_601/final)
	Target: x86_64-pc-linux-gnu
	Thread model: posix
```
It may not outperformance in your machine. You should modify the code if necessary.<br>
Hyper-threading should be carefully used. When it is enable, the latency and throughput increase together. (Low latency and high throughput are good.)<br>
2.<br>
Due to the first reason, I put a lot of other approaches in [intel_sad.cpp](https://github.com/Fdhvdu/SAD-512/blob/master/src/intel_sad.cpp).<br>
3.<br>
Write a program to test correctness. Compilers may do wrong optimization.<br>
I provide a check code in [test](https://github.com/Fdhvdu/SAD-512/tree/master/test). Compile the code by `clang++ -march=native -std=c++17 ...`.<br>
4.<br>
Latency and throughput is a good metrics, although they cannot perfectly express execution time sometimes.<br>
5.<br>
Although parameters are __m128i, __m256i and __m512i, you can use reinterpret_cast to convert integral type to destination type.<br>
However, when you use reinterpret_cast, you should check alignment first.
