#ifndef INTEL_SAD_H
#define INTEL_SAD_H
#include<cstdint>
#include<immintrin.h>

namespace nIntel
{
	//the following loadu functions are a single-pass function, they read successively
	//26.03 ns
	void loadu_64x8(const std::uint8_t *src,const std::int32_t *stride,__m512i *des);	//des[[0,13)]
	//45.68 ns
	void loadu_128x16(const std::uint8_t *src,const std::int32_t *stride,__m512i *des);	//des[[0,20)]
	//88.05 ns
	void loadu_256x32(const std::uint8_t *src,const std::int32_t *stride,__m512i *des);	//des[[0,48)]
	//172.38 ns
	void loadu_512x64(const std::uint8_t *src,const std::int32_t *stride,__m512i *des);	//des[[0,64)]

	void mul_stridex512x4(std::int32_t stride,__m512i *des);	//each type is int32_t
	void mul_stridex512x8(std::int32_t stride,__m512i *des);	//each type is int64_t
	
	//6.31 ns
	std::int32_t sad_512x1(const __m512i *a,const std::uint8_t *src,const std::int32_t *stride);
	//9.64 ns
	void sad_512x1(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::int32_t *stride,__m128i *des	//only the first 64 bit of des is useful
	);
	//13.50 ns
	void sad_512x1(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::int32_t *stride,__m128i *des
	);
	//24.31 ns
	void sad_512x1(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::uint8_t *cmp_4
		,const std::uint8_t *cmp_5
		,const std::uint8_t *cmp_6
		,const std::uint8_t *cmp_7
		,const std::int32_t *stride,__m256i *des
	);
	//10.72 ns
	std::int32_t sad_512x4(const __m512i *a,const std::uint8_t *src,const std::int32_t *stride);
	//14.06 ns
	void sad_512x4(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::int32_t *stride,__m128i *des	//only the first 64 bit of des is useful
	);
	//22.31 ns
	void sad_512x4(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::int32_t *stride,__m128i *des
	);
	//37.74 ns
	void sad_512x4(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::uint8_t *cmp_4
		,const std::uint8_t *cmp_5
		,const std::uint8_t *cmp_6
		,const std::uint8_t *cmp_7
		,const std::int32_t *stride,__m256i *des
	);
	//18.99 ns
	std::int32_t sad_512x16(const __m512i *a,const std::uint8_t *src,const std::int32_t *stride);
	//30.31 ns
	void sad_512x16(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::int32_t *stride,__m128i *des	//only the first 64 bit of des is useful
	);
	//46.65 ns
	void sad_512x16(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::int32_t *stride,__m128i *des
	);
	//82.93 ns
	void sad_512x16(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::uint8_t *cmp_4
		,const std::uint8_t *cmp_5
		,const std::uint8_t *cmp_6
		,const std::uint8_t *cmp_7
		,const std::int32_t *stride,__m256i *des
	);
	//59.37 ns
	std::int32_t sad_512x64(const __m512i *a,const std::uint8_t *src,const std::int32_t *stride);
	//87.84 ns
	void sad_512x64(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::int32_t *stride,__m128i *des	//only the first 64 bit of des is useful
	);
	//131.01 ns
	void sad_512x64(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::int32_t *stride,__m128i *des
	);
	//234.25 ns
	void sad_512x64(const __m512i *a
		,const std::uint8_t *cmp_0
		,const std::uint8_t *cmp_1
		,const std::uint8_t *cmp_2
		,const std::uint8_t *cmp_3
		,const std::uint8_t *cmp_4
		,const std::uint8_t *cmp_5
		,const std::uint8_t *cmp_6
		,const std::uint8_t *cmp_7
		,const std::int32_t *stride,__m256i *des
	);
}

#endif