#include"../header/intel_sad.hpp"
#include<type_traits>
#include<emmintrin.h>	//_mm_add_epi32
#include<immintrin.h>
#include<pmmintrin.h>	//_mm_loadu_si128
#include<smmintrin.h>	//_mm_extract_epi32
using namespace std;

#define MULTIPLE_SAD_BY_ADD(a0,b0,a1,b1)\
_mm512_add_epi32(_mm512_sad_epu8(a0,b0),_mm512_sad_epu8(a1,b1))

#define MULTIPLE_SAD(a0,b0,a1,b1)\
_mm512_mask_shuffle_epi32(_mm512_sad_epu8(a0,b0),mask16_1010101010101010,_mm512_sad_epu8(a1,b1),_MM_PERM_CDAB)

#define MULTIPLE_SAD_ADD_0(func,a0,b0,a1,b1,a2,b2,a3,b3)\
_mm512_add_epi32(func(a0,b0,a1,b1),func(a2,b2,a3,b3))
#define MULTIPLE_SAD_ADD_1(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7)\
_mm512_add_epi32(MULTIPLE_SAD_ADD_0(func,a0,b0,a1,b1,a2,b2,a3,b3),MULTIPLE_SAD_ADD_0(func,a4,b4,a5,b5,a6,b6,a7,b7))
#define MULTIPLE_SAD_ADD_2(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15)\
_mm512_add_epi32(MULTIPLE_SAD_ADD_1(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7),MULTIPLE_SAD_ADD_1(func,a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15))
#define MULTIPLE_SAD_ADD_3(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15,a16,b16,a17,b17,a18,b18,a19,b19,a20,b20,a21,b21,a22,b22,a23,b23,a24,b24,a25,b25,a26,b26,a27,b27,a28,b28,a29,b29,a30,b30,a31,b31)\
_mm512_add_epi32(MULTIPLE_SAD_ADD_2(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7,a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15),MULTIPLE_SAD_ADD_2(func,a16,b16,a17,b17,a18,b18,a19,b19,a20,b20,a21,b21,a22,b22,a23,b23,a24,b24,a25,b25,a26,b26,a27,b27,a28,b28,a29,b29,a30,b30,a31,b31))

#define MULTIPLE_SAD_ADD_0_ADD_THEN_SHUFFLE(func,a0,b0,a1,b1,a2,b2,a3,b3)\
_mm512_mask_shuffle_epi32(MULTIPLE_SAD_BY_ADD(a0,b0,a2,b2),mask16_1010101010101010,MULTIPLE_SAD_BY_ADD(a1,b1,a3,b3),_MM_PERM_CDAB)
#define MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(func,a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7)\
_mm512_mask_shuffle_epi32(MULTIPLE_SAD_ADD_0(MULTIPLE_SAD_BY_ADD,a0,b0,a2,b2,a4,b4,a6,b6),mask16_1010101010101010,MULTIPLE_SAD_ADD_0(MULTIPLE_SAD_BY_ADD,a1,b1,a3,b3,a5,b5,a7,b7),_MM_PERM_CDAB)

#define EXTRACT512_TO_128X4_ADD32(t512) _mm_add_epi32(_mm_add_epi32(_mm512_castsi512_si128(t512),_mm512_extracti64x2_epi64(t512,1)),_mm_add_epi32(_mm512_extracti64x2_epi64(t512,2),_mm512_extracti64x2_epi64(t512,3)))

#if 1
#define ADD_128_0_2(t128) _mm_cvtsi128_si32(t128)+_mm_extract_epi32(t128,2)
#else
//slower
#define ADD_128_0_2(t128) _mm_cvtsi128_si32(_mm_add_epi32(t128,_mm_shuffle_epi32(t128,_MM_PERM_BADC)))
#endif

#define READ_64(addr) *reinterpret_cast<const int64_t*>(addr)
#define LOADU_128(addr) _mm_loadu_si128(reinterpret_cast<const __m128i*>(addr))
#define LOADU_256(addr) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(addr))

#define LOADU_128_TO_512(add0,add1,add2,add3) _mm512_inserti64x4(_mm512_inserti64x2(_mm512_castsi128_si512(LOADU_128(add0)),LOADU_128(add1),1),_mm256_inserti64x2(_mm256_castsi128_si256(LOADU_128(add2)),LOADU_128(add3),1),1)
#define LOADU_256_TO_512(add0,add1) _mm512_inserti64x4(_mm512_castsi256_si512(LOADU_256(add0)),LOADU_256(add1),1)

#if 1
#define LOAD_64X8_512(stride,src) _mm512_i32gather_epi64(*reinterpret_cast<const __m256i*>(stride),src,1)
#else
//slower
#define LOAD_64X8_512(stride,src) _mm512_set_epi64(READ_64(src+stride[7]),READ_64(src+stride[6]),READ_64(src+stride[5]),READ_64(src+stride[4]),READ_64(src+stride[3]),READ_64(src+stride[2]),READ_64(src+stride[1]),READ_64(src))
#endif

namespace nIntel
{
	namespace
	{
		const auto mask16_1010101010101010(_mm512_int2mask(0b1010101010101010));
		const auto mask8_10101010(_mm_movepi16_mask(_mm_set_epi32(1<<31,1<<31,1<<31,1<<31)));
		const auto mask8_11001100(_mm_movepi16_mask(_mm_set_epi32((1<<31)|(1<<15),0,(1<<31)|(1<<15),0)));
		const auto mask8_11110000(_mm_movepi16_mask(_mm_set_epi32((1<<31)|(1<<15),(1<<31)|(1<<15),0,0)));
		
		const auto m512_epi64_13_12_5_4_9_8_1_0(_mm512_set_epi64(0b1'101,0b1'100,0b0'101,0b0'100,0b1'001,0b1'000,0b0'001,0b0'000));
		const auto m512_epi64_15_14_7_6_11_10_3_2(_mm512_set_epi64(0b1'111,0b1'110,0b01'11,0b0'110,0b1'011,0b1'010,0b0'011,0b0'010));
		const auto m512_epi64_5_4_1_0(_mm512_castsi256_si512(_mm256_set_epi64x(5,4,1,0)));
		const auto m512_epi64_7_6_3_2(_mm512_castsi256_si512(_mm256_set_epi64x(7,6,3,2)));
		const auto m512_epi32_14_12_10_8(_mm512_castsi128_si512(_mm_set_epi32(14,12,10,8)));
		const auto m512_epi32_6_4_2_0(_mm512_castsi128_si512(_mm_set_epi32(6,4,2,0)));
		const auto m512_epi32_12_8_4_0(_mm512_castsi128_si512(_mm_set_epi32(12,8,4,0)));
		const auto m512_epi32_14_10_6_2(_mm512_castsi128_si512(_mm_set_epi32(14,10,6,2)));
		const auto m512_epi32_14_12_10_8_6_4_2_0(_mm512_castsi256_si512(_mm256_set_epi32(14,12,10,8,6,4,2,0)));
		const auto m512_epi32_13_12_9_8_5_4_1_0(_mm512_castsi256_si512(_mm256_set_epi32(13,12,9,8,5,4,1,0)));
		const auto m512_epi32_15_14_11_10_7_6_3_2(_mm512_castsi256_si512(_mm256_set_epi32(15,14,11,10,7,6,3,2)));
	}
	
	void mul_stridex512x4(const int32_t stride,__m512i *des)
	{
		_mm512_store_si512(des,_mm512_mullo_epi32(_mm512_set1_epi32(stride),_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)));
		_mm512_store_si512(des+1,_mm512_mullo_epi32(_mm512_set1_epi32(stride),_mm512_set_epi32(31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16)));
		_mm512_store_si512(des+2,_mm512_mullo_epi32(_mm512_set1_epi32(stride),_mm512_set_epi32(47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32)));
		_mm512_store_si512(des+3,_mm512_mullo_epi32(_mm512_set1_epi32(stride),_mm512_set_epi32(63,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48)));
	}

	void mul_stridex512x8(const int32_t stride,__m512i *des)
	{
		_mm512_store_si512(des,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(7,6,5,4,3,2,1,0)));
		_mm512_store_si512(des+1,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(15,14,13,12,11,10,9,8)));
		_mm512_store_si512(des+2,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(23,22,21,20,19,18,17,16)));
		_mm512_store_si512(des+3,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(31,30,29,28,27,26,25,24)));
		_mm512_store_si512(des+4,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(39,38,37,36,35,34,33,32)));
		_mm512_store_si512(des+5,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(47,46,45,44,43,42,41,40)));
		_mm512_store_si512(des+6,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(55,54,53,52,51,50,49,48)));
		_mm512_store_si512(des+7,_mm512_mullo_epi64(_mm512_set1_epi64(stride),_mm512_set_epi64(63,62,61,60,59,58,57,56)));
	}

	void loadu_64x8(const uint8_t *src,const int32_t *stride,__m512i *des)
	{
		des[0]=_mm512_set1_epi64(READ_64(src));
		des[1]=_mm512_set1_epi64(READ_64(src+stride[1]));
		des[2]=_mm512_set1_epi64(READ_64(src+stride[2]));
		des[3]=_mm512_set1_epi64(READ_64(src+stride[3]));
		des[4]=_mm512_set1_epi64(READ_64(src+stride[4]));
		des[5]=_mm512_set1_epi64(READ_64(src+stride[5]));
		des[6]=_mm512_set1_epi64(READ_64(src+stride[6]));
		des[7]=_mm512_set1_epi64(READ_64(src+stride[7]));
		des[8]=_mm512_mask_blend_epi64(mask8_11110000,des[0],des[4]);
		des[9]=_mm512_mask_blend_epi64(mask8_11110000,des[1],des[5]);
		des[10]=_mm512_mask_blend_epi64(mask8_11110000,des[2],des[6]);
		des[11]=_mm512_mask_blend_epi64(mask8_11110000,des[3],des[7]);
		des[12]=_mm512_mask_blend_epi64(
			mask8_11001100
			,_mm512_mask_blend_epi64(mask8_10101010,des[8],des[9])
			,_mm512_mask_blend_epi64(mask8_10101010,des[10],des[11])
		);
	}
	
	void loadu_128x16(const uint8_t *src,const int32_t *stride,__m512i *des)
	{
		des[0]=_mm512_broadcast_i64x2(LOADU_128(src));
		des[1]=_mm512_broadcast_i64x2(LOADU_128(src+stride[1]));
		des[2]=_mm512_broadcast_i64x2(LOADU_128(src+stride[2]));
		des[3]=_mm512_broadcast_i64x2(LOADU_128(src+stride[3]));
		des[16]=_mm512_mask_blend_epi64(
			mask8_11110000
			,_mm512_mask_blend_epi64(mask8_11001100,des[0],des[1])
			,_mm512_mask_blend_epi64(mask8_11001100,des[2],des[3])
		);
		des[4]=_mm512_broadcast_i64x2(LOADU_128(src+stride[4]));
		des[5]=_mm512_broadcast_i64x2(LOADU_128(src+stride[5]));
		des[6]=_mm512_broadcast_i64x2(LOADU_128(src+stride[6]));
		des[7]=_mm512_broadcast_i64x2(LOADU_128(src+stride[7]));
		des[17]=_mm512_mask_blend_epi64(
			mask8_11110000
			,_mm512_mask_blend_epi64(mask8_11001100,des[4],des[5])
			,_mm512_mask_blend_epi64(mask8_11001100,des[6],des[7])
		);
		des[8]=_mm512_broadcast_i64x2(LOADU_128(src+stride[8]));
		des[9]=_mm512_broadcast_i64x2(LOADU_128(src+stride[9]));
		des[10]=_mm512_broadcast_i64x2(LOADU_128(src+stride[10]));
		des[11]=_mm512_broadcast_i64x2(LOADU_128(src+stride[11]));
		des[18]=_mm512_mask_blend_epi64(
			mask8_11110000
			,_mm512_mask_blend_epi64(mask8_11001100,des[8],des[9])
			,_mm512_mask_blend_epi64(mask8_11001100,des[10],des[11])
		);
		des[12]=_mm512_broadcast_i64x2(LOADU_128(src+stride[12]));
		des[13]=_mm512_broadcast_i64x2(LOADU_128(src+stride[13]));
		des[14]=_mm512_broadcast_i64x2(LOADU_128(src+stride[14]));
		des[15]=_mm512_broadcast_i64x2(LOADU_128(src+stride[15]));
		des[19]=_mm512_mask_blend_epi64(
			mask8_11110000
			,_mm512_mask_blend_epi64(mask8_11001100,des[12],des[13])
			,_mm512_mask_blend_epi64(mask8_11001100,des[14],des[15])
		);
	}

	void loadu_256x32(const uint8_t *src,const int32_t *stride,__m512i *des)
	{
		des[0]=_mm512_broadcast_i64x4(LOADU_256(src));
		des[1]=_mm512_broadcast_i64x4(LOADU_256(src+stride[1]));
		des[2]=_mm512_broadcast_i64x4(LOADU_256(src+stride[2]));
		des[3]=_mm512_broadcast_i64x4(LOADU_256(src+stride[3]));
		des[4]=_mm512_broadcast_i64x4(LOADU_256(src+stride[4]));
		des[5]=_mm512_broadcast_i64x4(LOADU_256(src+stride[5]));
		des[6]=_mm512_broadcast_i64x4(LOADU_256(src+stride[6]));
		des[7]=_mm512_broadcast_i64x4(LOADU_256(src+stride[7]));
		des[8]=_mm512_broadcast_i64x4(LOADU_256(src+stride[8]));
		des[9]=_mm512_broadcast_i64x4(LOADU_256(src+stride[9]));
		des[10]=_mm512_broadcast_i64x4(LOADU_256(src+stride[10]));
		des[11]=_mm512_broadcast_i64x4(LOADU_256(src+stride[11]));
		des[12]=_mm512_broadcast_i64x4(LOADU_256(src+stride[12]));
		des[13]=_mm512_broadcast_i64x4(LOADU_256(src+stride[13]));
		des[14]=_mm512_broadcast_i64x4(LOADU_256(src+stride[14]));
		des[15]=_mm512_broadcast_i64x4(LOADU_256(src+stride[15]));
		des[16]=_mm512_broadcast_i64x4(LOADU_256(src+stride[16]));
		des[17]=_mm512_broadcast_i64x4(LOADU_256(src+stride[17]));
		des[18]=_mm512_broadcast_i64x4(LOADU_256(src+stride[18]));
		des[19]=_mm512_broadcast_i64x4(LOADU_256(src+stride[19]));
		des[20]=_mm512_broadcast_i64x4(LOADU_256(src+stride[20]));
		des[21]=_mm512_broadcast_i64x4(LOADU_256(src+stride[21]));
		des[22]=_mm512_broadcast_i64x4(LOADU_256(src+stride[22]));
		des[23]=_mm512_broadcast_i64x4(LOADU_256(src+stride[23]));
		des[24]=_mm512_broadcast_i64x4(LOADU_256(src+stride[24]));
		des[25]=_mm512_broadcast_i64x4(LOADU_256(src+stride[25]));
		des[26]=_mm512_broadcast_i64x4(LOADU_256(src+stride[26]));
		des[27]=_mm512_broadcast_i64x4(LOADU_256(src+stride[27]));
		des[28]=_mm512_broadcast_i64x4(LOADU_256(src+stride[28]));
		des[29]=_mm512_broadcast_i64x4(LOADU_256(src+stride[29]));
		des[30]=_mm512_broadcast_i64x4(LOADU_256(src+stride[30]));
		des[31]=_mm512_broadcast_i64x4(LOADU_256(src+stride[31]));
		des[32]=_mm512_mask_blend_epi64(mask8_11110000,des[0],des[1]);
		des[33]=_mm512_mask_blend_epi64(mask8_11110000,des[2],des[3]);
		des[34]=_mm512_mask_blend_epi64(mask8_11110000,des[4],des[5]);
		des[35]=_mm512_mask_blend_epi64(mask8_11110000,des[6],des[7]);
		des[36]=_mm512_mask_blend_epi64(mask8_11110000,des[8],des[9]);
		des[37]=_mm512_mask_blend_epi64(mask8_11110000,des[10],des[11]);
		des[38]=_mm512_mask_blend_epi64(mask8_11110000,des[12],des[13]);
		des[39]=_mm512_mask_blend_epi64(mask8_11110000,des[14],des[15]);
		des[40]=_mm512_mask_blend_epi64(mask8_11110000,des[16],des[17]);
		des[41]=_mm512_mask_blend_epi64(mask8_11110000,des[18],des[19]);
		des[42]=_mm512_mask_blend_epi64(mask8_11110000,des[20],des[21]);
		des[43]=_mm512_mask_blend_epi64(mask8_11110000,des[22],des[23]);
		des[44]=_mm512_mask_blend_epi64(mask8_11110000,des[24],des[25]);
		des[45]=_mm512_mask_blend_epi64(mask8_11110000,des[26],des[27]);
		des[46]=_mm512_mask_blend_epi64(mask8_11110000,des[28],des[29]);
		des[47]=_mm512_mask_blend_epi64(mask8_11110000,des[30],des[31]);
	}

	void loadu_512x64(const uint8_t *src,const int32_t *stride,__m512i *des)
	{
		des[0]=_mm512_loadu_si512(src);
		des[1]=_mm512_loadu_si512(src+stride[1]);
		des[2]=_mm512_loadu_si512(src+stride[2]);
		des[3]=_mm512_loadu_si512(src+stride[3]);
		des[4]=_mm512_loadu_si512(src+stride[4]);
		des[5]=_mm512_loadu_si512(src+stride[5]);
		des[6]=_mm512_loadu_si512(src+stride[6]);
		des[7]=_mm512_loadu_si512(src+stride[7]);
		des[8]=_mm512_loadu_si512(src+stride[8]);
		des[9]=_mm512_loadu_si512(src+stride[9]);
		des[10]=_mm512_loadu_si512(src+stride[10]);
		des[11]=_mm512_loadu_si512(src+stride[11]);
		des[12]=_mm512_loadu_si512(src+stride[12]);
		des[13]=_mm512_loadu_si512(src+stride[13]);
		des[14]=_mm512_loadu_si512(src+stride[14]);
		des[15]=_mm512_loadu_si512(src+stride[15]);
		des[16]=_mm512_loadu_si512(src+stride[16]);
		des[17]=_mm512_loadu_si512(src+stride[17]);
		des[18]=_mm512_loadu_si512(src+stride[18]);
		des[19]=_mm512_loadu_si512(src+stride[19]);
		des[20]=_mm512_loadu_si512(src+stride[20]);
		des[21]=_mm512_loadu_si512(src+stride[21]);
		des[22]=_mm512_loadu_si512(src+stride[22]);
		des[23]=_mm512_loadu_si512(src+stride[23]);
		des[24]=_mm512_loadu_si512(src+stride[24]);
		des[25]=_mm512_loadu_si512(src+stride[25]);
		des[26]=_mm512_loadu_si512(src+stride[26]);
		des[27]=_mm512_loadu_si512(src+stride[27]);
		des[28]=_mm512_loadu_si512(src+stride[28]);
		des[29]=_mm512_loadu_si512(src+stride[29]);
		des[30]=_mm512_loadu_si512(src+stride[30]);
		des[31]=_mm512_loadu_si512(src+stride[31]);
		des[32]=_mm512_loadu_si512(src+stride[32]);
		des[33]=_mm512_loadu_si512(src+stride[33]);
		des[34]=_mm512_loadu_si512(src+stride[34]);
		des[35]=_mm512_loadu_si512(src+stride[35]);
		des[36]=_mm512_loadu_si512(src+stride[36]);
		des[37]=_mm512_loadu_si512(src+stride[37]);
		des[38]=_mm512_loadu_si512(src+stride[38]);
		des[39]=_mm512_loadu_si512(src+stride[39]);
		des[40]=_mm512_loadu_si512(src+stride[40]);
		des[41]=_mm512_loadu_si512(src+stride[41]);
		des[42]=_mm512_loadu_si512(src+stride[42]);
		des[43]=_mm512_loadu_si512(src+stride[43]);
		des[44]=_mm512_loadu_si512(src+stride[44]);
		des[45]=_mm512_loadu_si512(src+stride[45]);
		des[46]=_mm512_loadu_si512(src+stride[46]);
		des[47]=_mm512_loadu_si512(src+stride[47]);
		des[48]=_mm512_loadu_si512(src+stride[48]);
		des[49]=_mm512_loadu_si512(src+stride[49]);
		des[50]=_mm512_loadu_si512(src+stride[50]);
		des[51]=_mm512_loadu_si512(src+stride[51]);
		des[52]=_mm512_loadu_si512(src+stride[52]);
		des[53]=_mm512_loadu_si512(src+stride[53]);
		des[54]=_mm512_loadu_si512(src+stride[54]);
		des[55]=_mm512_loadu_si512(src+stride[55]);
		des[56]=_mm512_loadu_si512(src+stride[56]);
		des[57]=_mm512_loadu_si512(src+stride[57]);
		des[58]=_mm512_loadu_si512(src+stride[58]);
		des[59]=_mm512_loadu_si512(src+stride[59]);
		des[60]=_mm512_loadu_si512(src+stride[60]);
		des[61]=_mm512_loadu_si512(src+stride[61]);
		des[62]=_mm512_loadu_si512(src+stride[62]);
		des[63]=_mm512_loadu_si512(src+stride[63]);
	}

	int32_t sad_512x1(const __m512i *a,const uint8_t *src,const int32_t *stride)
	{
		a+=12;
		__m512i t512(_mm512_sad_epu8(*a,LOAD_64X8_512(stride,src)));
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		return ADD_128_0_2(t128);
	}
	
	void sad_512x1(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const int32_t *stride,__m128i *des
	)
	{
		a+=12;
		__m512i t512(MULTIPLE_SAD(
			a[0],LOAD_64X8_512(stride,cmp_0)
			,a[0],LOAD_64X8_512(stride,cmp_1)
		));
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		*des=_mm_add_epi32(t128,_mm_shuffle_epi32(t128,_MM_PERM_BADC));
	}
	
	void sad_512x1(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const int32_t *stride,__m128i *des
	)
	{
		a+=8;
		__m512i t512(MULTIPLE_SAD_ADD_0(
			MULTIPLE_SAD_BY_ADD
			,a[0],_mm512_set_epi64(
				READ_64(cmp_3+stride[4]),READ_64(cmp_2+stride[4]),READ_64(cmp_1+stride[4]),READ_64(cmp_0+stride[4])
				,READ_64(cmp_3),READ_64(cmp_2),READ_64(cmp_1),READ_64(cmp_0)
			)
			,a[1],_mm512_set_epi64(
				READ_64(cmp_3+stride[5]),READ_64(cmp_2+stride[5]),READ_64(cmp_1+stride[5]),READ_64(cmp_0+stride[5])
				,READ_64(cmp_3+stride[1]),READ_64(cmp_2+stride[1]),READ_64(cmp_1+stride[1]),READ_64(cmp_0+stride[1])
			)
			,a[2],_mm512_set_epi64(
				READ_64(cmp_3+stride[6]),READ_64(cmp_2+stride[6]),READ_64(cmp_1+stride[6]),READ_64(cmp_0+stride[6])
				,READ_64(cmp_3+stride[2]),READ_64(cmp_2+stride[2]),READ_64(cmp_1+stride[2]),READ_64(cmp_0+stride[2])
			)
			,a[3],_mm512_set_epi64(
				READ_64(cmp_3+stride[7]),READ_64(cmp_2+stride[7]),READ_64(cmp_1+stride[7]),READ_64(cmp_0+stride[7])
				,READ_64(cmp_3+stride[3]),READ_64(cmp_2+stride[3]),READ_64(cmp_1+stride[3]),READ_64(cmp_0+stride[3])
			)
		));
#if 1
		//1.68 ns
		*des=_mm256_castsi256_si128(
			_mm256_permutexvar_epi32(_mm512_castsi512_si256(m512_epi32_6_4_2_0)
				,_mm256_add_epi32(_mm512_castsi512_si256(t512),_mm512_extracti64x4_epi64(t512,1)))
		);
#else
		//1.96 ns
		*des=_mm_add_epi32(
			_mm512_castsi512_si128(_mm512_permutexvar_epi32(m512_epi32_14_12_10_8,t512))
			,_mm512_castsi512_si128(_mm512_permutexvar_epi32(m512_epi32_6_4_2_0,t512))
		);
#endif
	}

	void sad_512x1(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const uint8_t *cmp_4,const uint8_t *cmp_5
		,const uint8_t *cmp_6,const uint8_t *cmp_7
		,const int32_t *stride,__m256i *des
	)
	{
#if 1
		*des=_mm512_castsi512_si256(_mm512_permutexvar_epi32(m512_epi32_14_12_10_8_6_4_2_0,
			MULTIPLE_SAD_ADD_1(
				MULTIPLE_SAD_BY_ADD
				,a[0],_mm512_set_epi64(
					READ_64(cmp_7),READ_64(cmp_6),READ_64(cmp_5),READ_64(cmp_4)
					,READ_64(cmp_3),READ_64(cmp_2),READ_64(cmp_1),READ_64(cmp_0)
				)
				,a[1],_mm512_set_epi64(
					READ_64(cmp_7+stride[1]),READ_64(cmp_6+stride[1]),READ_64(cmp_5+stride[1]),READ_64(cmp_4+stride[1])
					,READ_64(cmp_3+stride[1]),READ_64(cmp_2+stride[1]),READ_64(cmp_1+stride[1]),READ_64(cmp_0+stride[1])
				)
				,a[2],_mm512_set_epi64(
					READ_64(cmp_7+stride[2]),READ_64(cmp_6+stride[2]),READ_64(cmp_5+stride[2]),READ_64(cmp_4+stride[2])
					,READ_64(cmp_3+stride[2]),READ_64(cmp_2+stride[2]),READ_64(cmp_1+stride[2]),READ_64(cmp_0+stride[2])
				)
				,a[3],_mm512_set_epi64(
					READ_64(cmp_7+stride[3]),READ_64(cmp_6+stride[3]),READ_64(cmp_5+stride[3]),READ_64(cmp_4+stride[3])
					,READ_64(cmp_3+stride[3]),READ_64(cmp_2+stride[3]),READ_64(cmp_1+stride[3]),READ_64(cmp_0+stride[3])
				)
				,a[4],_mm512_set_epi64(
					READ_64(cmp_7+stride[4]),READ_64(cmp_6+stride[4]),READ_64(cmp_5+stride[4]),READ_64(cmp_4+stride[4])
					,READ_64(cmp_3+stride[4]),READ_64(cmp_2+stride[4]),READ_64(cmp_1+stride[4]),READ_64(cmp_0+stride[4])
				)
				,a[5],_mm512_set_epi64(
					READ_64(cmp_7+stride[5]),READ_64(cmp_6+stride[5]),READ_64(cmp_5+stride[5]),READ_64(cmp_4+stride[5])
					,READ_64(cmp_3+stride[5]),READ_64(cmp_2+stride[5]),READ_64(cmp_1+stride[5]),READ_64(cmp_0+stride[5])
				)
				,a[6],_mm512_set_epi64(
					READ_64(cmp_7+stride[6]),READ_64(cmp_6+stride[6]),READ_64(cmp_5+stride[6]),READ_64(cmp_4+stride[6])
					,READ_64(cmp_3+stride[6]),READ_64(cmp_2+stride[6]),READ_64(cmp_1+stride[6]),READ_64(cmp_0+stride[6])
				)
				,a[7],_mm512_set_epi64(
					READ_64(cmp_7+stride[7]),READ_64(cmp_6+stride[7]),READ_64(cmp_5+stride[7]),READ_64(cmp_4+stride[7])
					,READ_64(cmp_3+stride[7]),READ_64(cmp_2+stride[7]),READ_64(cmp_1+stride[7]),READ_64(cmp_0+stride[7])
				)
			)
		));
#else
		//slower
		__m512i t512(_mm512_add_epi32(
			_mm512_add_epi32(
				MULTIPLE_SAD(
					a[0],_mm512_set_epi64(
						READ_64(cmp_7),READ_64(cmp_6),READ_64(cmp_5),READ_64(cmp_4)
						,READ_64(cmp_3),READ_64(cmp_2),READ_64(cmp_1),READ_64(cmp_0)
					)
					,a[1],_mm512_set_epi64(
						READ_64(cmp_7+stride[1]),READ_64(cmp_6+stride[1]),READ_64(cmp_5+stride[1]),READ_64(cmp_4+stride[1])
						,READ_64(cmp_3+stride[1]),READ_64(cmp_2+stride[1]),READ_64(cmp_1+stride[1]),READ_64(cmp_0+stride[1])
					)
				)
				,
				MULTIPLE_SAD(
					a[2],_mm512_set_epi64(
						READ_64(cmp_7+stride[2]),READ_64(cmp_6+stride[2]),READ_64(cmp_5+stride[2]),READ_64(cmp_4+stride[2])
						,READ_64(cmp_3+stride[2]),READ_64(cmp_2+stride[2]),READ_64(cmp_1+stride[2]),READ_64(cmp_0+stride[2])
					)
					,a[3],_mm512_set_epi64(
						READ_64(cmp_7+stride[3]),READ_64(cmp_6+stride[3]),READ_64(cmp_5+stride[3]),READ_64(cmp_4+stride[3])
						,READ_64(cmp_3+stride[3]),READ_64(cmp_2+stride[3]),READ_64(cmp_1+stride[3]),READ_64(cmp_0+stride[3])
					)
				)
			),_mm512_add_epi32(
				MULTIPLE_SAD(
					a[4],_mm512_set_epi64(
						READ_64(cmp_7+stride[4]),READ_64(cmp_6+stride[4]),READ_64(cmp_5+stride[4]),READ_64(cmp_4+stride[4])
						,READ_64(cmp_3+stride[4]),READ_64(cmp_2+stride[4]),READ_64(cmp_1+stride[4]),READ_64(cmp_0+stride[4])
					)
					,a[5],_mm512_set_epi64(
						READ_64(cmp_7+stride[5]),READ_64(cmp_6+stride[5]),READ_64(cmp_5+stride[5]),READ_64(cmp_4+stride[5])
						,READ_64(cmp_3+stride[5]),READ_64(cmp_2+stride[5]),READ_64(cmp_1+stride[5]),READ_64(cmp_0+stride[5])
					)
				)
				,MULTIPLE_SAD(
					a[6],_mm512_set_epi64(
						READ_64(cmp_7+stride[6]),READ_64(cmp_6+stride[6]),READ_64(cmp_5+stride[6]),READ_64(cmp_4+stride[6])
						,READ_64(cmp_3+stride[6]),READ_64(cmp_2+stride[6]),READ_64(cmp_1+stride[6]),READ_64(cmp_0+stride[6])
					)
					,a[7],_mm512_set_epi64(
						READ_64(cmp_7+stride[7]),READ_64(cmp_6+stride[7]),READ_64(cmp_5+stride[7]),READ_64(cmp_4+stride[7])
						,READ_64(cmp_3+stride[7]),READ_64(cmp_2+stride[7]),READ_64(cmp_1+stride[7]),READ_64(cmp_0+stride[7])
					)
				)
			)));
		*des=_mm512_castsi512_si256(_mm512_permutexvar_epi32(m512_epi32_14_12_10_8_6_4_2_0,_mm512_add_epi32(t512,_mm512_shuffle_epi32(t512,_MM_PERM_CDAB))));
#endif
	}

	int32_t sad_512x4(const __m512i *a,const uint8_t *src,const int32_t *stride)
	{
		a+=16;
		__m512i t512(
			MULTIPLE_SAD_ADD_0(
				MULTIPLE_SAD_BY_ADD
				,a[0],LOADU_128_TO_512(src,src+stride[1],src+stride[2],src+stride[3])
				,a[1],LOADU_128_TO_512(src+stride[4],src+stride[5],src+stride[6],src+stride[7])
				,a[2],LOADU_128_TO_512(src+stride[8],src+stride[9],src+stride[10],src+stride[11])
				,a[3],LOADU_128_TO_512(src+stride[12],src+stride[13],src+stride[14],src+stride[15])
			)
		);
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		return ADD_128_0_2(t128);
	}

	void sad_512x4(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const int32_t *stride,__m128i *des
	)
	{
		a+=16;
#if 1
		__m512i t512(MULTIPLE_SAD_ADD_1(
			MULTIPLE_SAD
			,a[0],LOADU_128_TO_512(cmp_0,cmp_0+stride[1],cmp_0+stride[2],cmp_0+stride[3])
			,a[0],LOADU_128_TO_512(cmp_1,cmp_1+stride[1],cmp_1+stride[2],cmp_1+stride[3])
			,a[1],LOADU_128_TO_512(cmp_0+stride[4],cmp_0+stride[5],cmp_0+stride[6],cmp_0+stride[7])
			,a[1],LOADU_128_TO_512(cmp_1+stride[4],cmp_1+stride[5],cmp_1+stride[6],cmp_1+stride[7])
			,a[2],LOADU_128_TO_512(cmp_0+stride[8],cmp_0+stride[9],cmp_0+stride[10],cmp_0+stride[11])
			,a[2],LOADU_128_TO_512(cmp_1+stride[8],cmp_1+stride[9],cmp_1+stride[10],cmp_1+stride[11])
			,a[3],LOADU_128_TO_512(cmp_0+stride[12],cmp_0+stride[13],cmp_0+stride[14],cmp_0+stride[15])
			,a[3],LOADU_128_TO_512(cmp_1+stride[12],cmp_1+stride[13],cmp_1+stride[14],cmp_1+stride[15])
		));
#else
		//slower
		__m512i t512(MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_128_TO_512(cmp_0,cmp_0+stride[1],cmp_0+stride[2],cmp_0+stride[3])
			,a[0],LOADU_128_TO_512(cmp_1,cmp_1+stride[1],cmp_1+stride[2],cmp_1+stride[3])
			,a[1],LOADU_128_TO_512(cmp_0+stride[4],cmp_0+stride[5],cmp_0+stride[6],cmp_0+stride[7])
			,a[1],LOADU_128_TO_512(cmp_1+stride[4],cmp_1+stride[5],cmp_1+stride[6],cmp_1+stride[7])
			,a[2],LOADU_128_TO_512(cmp_0+stride[8],cmp_0+stride[9],cmp_0+stride[10],cmp_0+stride[11])
			,a[2],LOADU_128_TO_512(cmp_1+stride[8],cmp_1+stride[9],cmp_1+stride[10],cmp_1+stride[11])
			,a[3],LOADU_128_TO_512(cmp_0+stride[12],cmp_0+stride[13],cmp_0+stride[14],cmp_0+stride[15])
			,a[3],LOADU_128_TO_512(cmp_1+stride[12],cmp_1+stride[13],cmp_1+stride[14],cmp_1+stride[15])
		));
#endif
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		*des=_mm_add_epi32(t128,_mm_shuffle_epi32(t128,_MM_PERM_BADC));
	}

	void sad_512x4(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const int32_t *stride,__m128i *des
	)
	{
		__m512i t512(MULTIPLE_SAD_ADD_2(
			MULTIPLE_SAD_BY_ADD
			,a[0],LOADU_128_TO_512(cmp_0,cmp_1,cmp_2,cmp_3)
			,a[1],LOADU_128_TO_512(cmp_0+stride[1],cmp_1+stride[1],cmp_2+stride[1],cmp_3+stride[1])
			,a[2],LOADU_128_TO_512(cmp_0+stride[2],cmp_1+stride[2],cmp_2+stride[2],cmp_3+stride[2])
			,a[3],LOADU_128_TO_512(cmp_0+stride[3],cmp_1+stride[3],cmp_2+stride[3],cmp_3+stride[3])
			,a[4],LOADU_128_TO_512(cmp_0+stride[4],cmp_1+stride[4],cmp_2+stride[4],cmp_3+stride[4])
			,a[5],LOADU_128_TO_512(cmp_0+stride[5],cmp_1+stride[5],cmp_2+stride[5],cmp_3+stride[5])
			,a[6],LOADU_128_TO_512(cmp_0+stride[6],cmp_1+stride[6],cmp_2+stride[6],cmp_3+stride[6])
			,a[7],LOADU_128_TO_512(cmp_0+stride[7],cmp_1+stride[7],cmp_2+stride[7],cmp_3+stride[7])
			,a[8],LOADU_128_TO_512(cmp_0+stride[8],cmp_1+stride[8],cmp_2+stride[8],cmp_3+stride[8])
			,a[9],LOADU_128_TO_512(cmp_0+stride[9],cmp_1+stride[9],cmp_2+stride[9],cmp_3+stride[9])
			,a[10],LOADU_128_TO_512(cmp_0+stride[10],cmp_1+stride[10],cmp_2+stride[10],cmp_3+stride[10])
			,a[11],LOADU_128_TO_512(cmp_0+stride[11],cmp_1+stride[11],cmp_2+stride[11],cmp_3+stride[11])
			,a[12],LOADU_128_TO_512(cmp_0+stride[12],cmp_1+stride[12],cmp_2+stride[12],cmp_3+stride[12])
			,a[13],LOADU_128_TO_512(cmp_0+stride[13],cmp_1+stride[13],cmp_2+stride[13],cmp_3+stride[13])
			,a[14],LOADU_128_TO_512(cmp_0+stride[14],cmp_1+stride[14],cmp_2+stride[14],cmp_3+stride[14])
			,a[15],LOADU_128_TO_512(cmp_0+stride[15],cmp_1+stride[15],cmp_2+stride[15],cmp_3+stride[15])
		));
#if 1
		//1.68 ns
		*des=_mm512_castsi512_si128(
			_mm512_permutexvar_epi32(
				m512_epi32_12_8_4_0
				,_mm512_add_epi32(t512,_mm512_shuffle_epi32(t512,_MM_PERM_BADC))
			)
		);
#else
		//1.96 ns
		*des=_mm_add_epi32(
			_mm512_castsi512_si128(_mm512_permutexvar_epi32(m512_epi32_12_8_4_0,t512))
			,_mm512_castsi512_si128(_mm512_permutexvar_epi32(m512_epi32_14_10_6_2,t512))
		);
#endif
	}

	void sad_512x4(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const uint8_t *cmp_4,const uint8_t *cmp_5
		,const uint8_t *cmp_6,const uint8_t *cmp_7
		,const int32_t *stride,__m256i *des
	)
	{
#if 1
		__m512i t512(
			_mm512_add_epi32(
				_mm512_add_epi32(
					MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
						MULTIPLE_SAD_BY_ADD
						,a[0],LOADU_128_TO_512(cmp_0,cmp_2,cmp_4,cmp_6)
						,a[0],LOADU_128_TO_512(cmp_1,cmp_3,cmp_5,cmp_7)
						,a[1],LOADU_128_TO_512(cmp_0+stride[1],cmp_2+stride[1],cmp_4+stride[1],cmp_6+stride[1])
						,a[1],LOADU_128_TO_512(cmp_1+stride[1],cmp_3+stride[1],cmp_5+stride[1],cmp_7+stride[1])
						,a[2],LOADU_128_TO_512(cmp_0+stride[2],cmp_2+stride[2],cmp_4+stride[2],cmp_6+stride[2])
						,a[2],LOADU_128_TO_512(cmp_1+stride[2],cmp_3+stride[2],cmp_5+stride[2],cmp_7+stride[2])
						,a[3],LOADU_128_TO_512(cmp_0+stride[3],cmp_2+stride[3],cmp_4+stride[3],cmp_6+stride[3])
						,a[3],LOADU_128_TO_512(cmp_1+stride[3],cmp_3+stride[3],cmp_5+stride[3],cmp_7+stride[3])
					)
					,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
						MULTIPLE_SAD_BY_ADD
						,a[4],LOADU_128_TO_512(cmp_0+stride[4],cmp_2+stride[4],cmp_4+stride[4],cmp_6+stride[4])
						,a[4],LOADU_128_TO_512(cmp_1+stride[4],cmp_3+stride[4],cmp_5+stride[4],cmp_7+stride[4])
						,a[5],LOADU_128_TO_512(cmp_0+stride[5],cmp_2+stride[5],cmp_4+stride[5],cmp_6+stride[5])
						,a[5],LOADU_128_TO_512(cmp_1+stride[5],cmp_3+stride[5],cmp_5+stride[5],cmp_7+stride[5])
						,a[6],LOADU_128_TO_512(cmp_0+stride[6],cmp_2+stride[6],cmp_4+stride[6],cmp_6+stride[6])
						,a[6],LOADU_128_TO_512(cmp_1+stride[6],cmp_3+stride[6],cmp_5+stride[6],cmp_7+stride[6])
						,a[7],LOADU_128_TO_512(cmp_0+stride[7],cmp_2+stride[7],cmp_4+stride[7],cmp_6+stride[7])
						,a[7],LOADU_128_TO_512(cmp_1+stride[7],cmp_3+stride[7],cmp_5+stride[7],cmp_7+stride[7])
					)
				)
				,_mm512_add_epi32(
					MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
						MULTIPLE_SAD_BY_ADD
						,a[8],LOADU_128_TO_512(cmp_0+stride[8],cmp_2+stride[8],cmp_4+stride[8],cmp_6+stride[8])
						,a[8],LOADU_128_TO_512(cmp_1+stride[8],cmp_3+stride[8],cmp_5+stride[8],cmp_7+stride[8])
						,a[9],LOADU_128_TO_512(cmp_0+stride[9],cmp_2+stride[9],cmp_4+stride[9],cmp_6+stride[9])
						,a[9],LOADU_128_TO_512(cmp_1+stride[9],cmp_3+stride[9],cmp_5+stride[9],cmp_7+stride[9])
						,a[10],LOADU_128_TO_512(cmp_0+stride[10],cmp_2+stride[10],cmp_4+stride[10],cmp_6+stride[10])
						,a[10],LOADU_128_TO_512(cmp_1+stride[10],cmp_3+stride[10],cmp_5+stride[10],cmp_7+stride[10])
						,a[11],LOADU_128_TO_512(cmp_0+stride[11],cmp_2+stride[11],cmp_4+stride[11],cmp_6+stride[11])
						,a[11],LOADU_128_TO_512(cmp_1+stride[11],cmp_3+stride[11],cmp_5+stride[11],cmp_7+stride[11])
					)
					,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
						MULTIPLE_SAD_BY_ADD
						,a[12],LOADU_128_TO_512(cmp_0+stride[12],cmp_2+stride[12],cmp_4+stride[12],cmp_6+stride[12])
						,a[12],LOADU_128_TO_512(cmp_1+stride[12],cmp_3+stride[12],cmp_5+stride[12],cmp_7+stride[12])
						,a[13],LOADU_128_TO_512(cmp_0+stride[13],cmp_2+stride[13],cmp_4+stride[13],cmp_6+stride[13])
						,a[13],LOADU_128_TO_512(cmp_1+stride[13],cmp_3+stride[13],cmp_5+stride[13],cmp_7+stride[13])
						,a[14],LOADU_128_TO_512(cmp_0+stride[14],cmp_2+stride[14],cmp_4+stride[14],cmp_6+stride[14])
						,a[14],LOADU_128_TO_512(cmp_1+stride[14],cmp_3+stride[14],cmp_5+stride[14],cmp_7+stride[14])
						,a[15],LOADU_128_TO_512(cmp_0+stride[15],cmp_2+stride[15],cmp_4+stride[15],cmp_6+stride[15])
						,a[15],LOADU_128_TO_512(cmp_1+stride[15],cmp_3+stride[15],cmp_5+stride[15],cmp_7+stride[15])
					)
				)
			)
		);
#else
		//slower
		__m512i t512(MULTIPLE_SAD_ADD_3(
			MULTIPLE_SAD
			,a[0],LOADU_128_TO_512(cmp_0,cmp_2,cmp_4,cmp_6)
			,a[0],LOADU_128_TO_512(cmp_1,cmp_3,cmp_5,cmp_7)
			,a[1],LOADU_128_TO_512(cmp_0+stride[1],cmp_2+stride[1],cmp_4+stride[1],cmp_6+stride[1])
			,a[1],LOADU_128_TO_512(cmp_1+stride[1],cmp_3+stride[1],cmp_5+stride[1],cmp_7+stride[1])
			,a[2],LOADU_128_TO_512(cmp_0+stride[2],cmp_2+stride[2],cmp_4+stride[2],cmp_6+stride[2])
			,a[2],LOADU_128_TO_512(cmp_1+stride[2],cmp_3+stride[2],cmp_5+stride[2],cmp_7+stride[2])
			,a[3],LOADU_128_TO_512(cmp_0+stride[3],cmp_2+stride[3],cmp_4+stride[3],cmp_6+stride[3])
			,a[3],LOADU_128_TO_512(cmp_1+stride[3],cmp_3+stride[3],cmp_5+stride[3],cmp_7+stride[3])
			,a[4],LOADU_128_TO_512(cmp_0+stride[4],cmp_2+stride[4],cmp_4+stride[4],cmp_6+stride[4])
			,a[4],LOADU_128_TO_512(cmp_1+stride[4],cmp_3+stride[4],cmp_5+stride[4],cmp_7+stride[4])
			,a[5],LOADU_128_TO_512(cmp_0+stride[5],cmp_2+stride[5],cmp_4+stride[5],cmp_6+stride[5])
			,a[5],LOADU_128_TO_512(cmp_1+stride[5],cmp_3+stride[5],cmp_5+stride[5],cmp_7+stride[5])
			,a[6],LOADU_128_TO_512(cmp_0+stride[6],cmp_2+stride[6],cmp_4+stride[6],cmp_6+stride[6])
			,a[6],LOADU_128_TO_512(cmp_1+stride[6],cmp_3+stride[6],cmp_5+stride[6],cmp_7+stride[6])
			,a[7],LOADU_128_TO_512(cmp_0+stride[7],cmp_2+stride[7],cmp_4+stride[7],cmp_6+stride[7])
			,a[7],LOADU_128_TO_512(cmp_1+stride[7],cmp_3+stride[7],cmp_5+stride[7],cmp_7+stride[7])
			,a[8],LOADU_128_TO_512(cmp_0+stride[8],cmp_2+stride[8],cmp_4+stride[8],cmp_6+stride[8])
			,a[8],LOADU_128_TO_512(cmp_1+stride[8],cmp_3+stride[8],cmp_5+stride[8],cmp_7+stride[8])
			,a[9],LOADU_128_TO_512(cmp_0+stride[9],cmp_2+stride[9],cmp_4+stride[9],cmp_6+stride[9])
			,a[9],LOADU_128_TO_512(cmp_1+stride[9],cmp_3+stride[9],cmp_5+stride[9],cmp_7+stride[9])
			,a[10],LOADU_128_TO_512(cmp_0+stride[10],cmp_2+stride[10],cmp_4+stride[10],cmp_6+stride[10])
			,a[10],LOADU_128_TO_512(cmp_1+stride[10],cmp_3+stride[10],cmp_5+stride[10],cmp_7+stride[10])
			,a[11],LOADU_128_TO_512(cmp_0+stride[11],cmp_2+stride[11],cmp_4+stride[11],cmp_6+stride[11])
			,a[11],LOADU_128_TO_512(cmp_1+stride[11],cmp_3+stride[11],cmp_5+stride[11],cmp_7+stride[11])
			,a[12],LOADU_128_TO_512(cmp_0+stride[12],cmp_2+stride[12],cmp_4+stride[12],cmp_6+stride[12])
			,a[12],LOADU_128_TO_512(cmp_1+stride[12],cmp_3+stride[12],cmp_5+stride[12],cmp_7+stride[12])
			,a[13],LOADU_128_TO_512(cmp_0+stride[13],cmp_2+stride[13],cmp_4+stride[13],cmp_6+stride[13])
			,a[13],LOADU_128_TO_512(cmp_1+stride[13],cmp_3+stride[13],cmp_5+stride[13],cmp_7+stride[13])
			,a[14],LOADU_128_TO_512(cmp_0+stride[14],cmp_2+stride[14],cmp_4+stride[14],cmp_6+stride[14])
			,a[14],LOADU_128_TO_512(cmp_1+stride[14],cmp_3+stride[14],cmp_5+stride[14],cmp_7+stride[14])
			,a[15],LOADU_128_TO_512(cmp_0+stride[15],cmp_2+stride[15],cmp_4+stride[15],cmp_6+stride[15])
			,a[15],LOADU_128_TO_512(cmp_1+stride[15],cmp_3+stride[15],cmp_5+stride[15],cmp_7+stride[15])
		));
#endif
#if 1	//they have almost same performance
		*des=_mm512_castsi512_si256(
			_mm512_permutexvar_epi32(
				m512_epi32_13_12_9_8_5_4_1_0
				,_mm512_add_epi32(t512,_mm512_shuffle_epi32(t512,_MM_PERM_BADC))
			)
		);
#else
		*des=_mm256_add_epi32(
			_mm512_castsi512_si256(_mm512_permutexvar_epi32(m512_epi32_13_12_9_8_5_4_1_0,t512))
			,_mm512_castsi512_si256(_mm512_permutexvar_epi32(m512_epi32_15_14_11_10_7_6_3_2,t512))
		);
#endif
	}

	int32_t sad_512x16(const __m512i *a,const uint8_t *src,const int32_t *stride)
	{
		a+=32;
		__m512i t512(
			MULTIPLE_SAD_ADD_2(
				MULTIPLE_SAD_BY_ADD
				,a[0],LOADU_256_TO_512(src,src+stride[1])
				,a[1],LOADU_256_TO_512(src+stride[2],src+stride[3])
				,a[2],LOADU_256_TO_512(src+stride[4],src+stride[5])
				,a[3],LOADU_256_TO_512(src+stride[6],src+stride[7])
				,a[4],LOADU_256_TO_512(src+stride[8],src+stride[9])
				,a[5],LOADU_256_TO_512(src+stride[10],src+stride[11])
				,a[6],LOADU_256_TO_512(src+stride[12],src+stride[13])
				,a[7],LOADU_256_TO_512(src+stride[14],src+stride[15])
				,a[8],LOADU_256_TO_512(src+stride[16],src+stride[17])
				,a[9],LOADU_256_TO_512(src+stride[18],src+stride[19])
				,a[10],LOADU_256_TO_512(src+stride[20],src+stride[21])
				,a[11],LOADU_256_TO_512(src+stride[22],src+stride[23])
				,a[12],LOADU_256_TO_512(src+stride[24],src+stride[25])
				,a[13],LOADU_256_TO_512(src+stride[26],src+stride[27])
				,a[14],LOADU_256_TO_512(src+stride[28],src+stride[29])
				,a[15],LOADU_256_TO_512(src+stride[30],src+stride[31])
			)
		);
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		return ADD_128_0_2(t128);
	}

	void sad_512x16(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const int32_t *stride,__m128i *des
	)
	{
		a+=32;
		__m512i t512(MULTIPLE_SAD_ADD_0_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_256_TO_512(cmp_0,cmp_0+stride[1]),a[0],LOADU_256_TO_512(cmp_1,cmp_1+stride[1])
			,a[1],LOADU_256_TO_512(cmp_0+stride[2],cmp_0+stride[3]),a[1],LOADU_256_TO_512(cmp_1+stride[2],cmp_1+stride[3])
		));
		for(remove_cv_t<decltype(32)> i(4),j(2);i!=32;i+=4,j+=2)
			t512=_mm512_add_epi32(t512
				,MULTIPLE_SAD_ADD_0_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[j],LOADU_256_TO_512(cmp_0+stride[i],cmp_0+stride[i+1]),a[j],LOADU_256_TO_512(cmp_1+stride[i],cmp_1+stride[i+1])
					,a[j+1],LOADU_256_TO_512(cmp_0+stride[i+2],cmp_0+stride[i+3]),a[j+1],LOADU_256_TO_512(cmp_1+stride[i+2],cmp_1+stride[i+3])
				)
			);
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		*des=_mm_add_epi32(t128,_mm_shuffle_epi32(t128,_MM_PERM_BADC));
	}

	void sad_512x16(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const int32_t *stride,__m128i *des
	)
	{
		a+=32;
		__m512i t512[2];
		t512[0]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_256_TO_512(cmp_0,cmp_0+stride[1]),a[0],LOADU_256_TO_512(cmp_1,cmp_1+stride[1])
			,a[1],LOADU_256_TO_512(cmp_0+stride[2],cmp_0+stride[3]),a[1],LOADU_256_TO_512(cmp_1+stride[2],cmp_1+stride[3])
			,a[2],LOADU_256_TO_512(cmp_0+stride[4],cmp_0+stride[5]),a[2],LOADU_256_TO_512(cmp_1+stride[4],cmp_1+stride[5])
			,a[3],LOADU_256_TO_512(cmp_0+stride[6],cmp_0+stride[7]),a[3],LOADU_256_TO_512(cmp_1+stride[6],cmp_1+stride[7])
		);
		t512[1]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_256_TO_512(cmp_2,cmp_2+stride[1]),a[0],LOADU_256_TO_512(cmp_3,cmp_3+stride[1])
			,a[1],LOADU_256_TO_512(cmp_2+stride[2],cmp_2+stride[3]),a[1],LOADU_256_TO_512(cmp_3+stride[2],cmp_3+stride[3])
			,a[2],LOADU_256_TO_512(cmp_2+stride[4],cmp_2+stride[5]),a[2],LOADU_256_TO_512(cmp_3+stride[4],cmp_3+stride[5])
			,a[3],LOADU_256_TO_512(cmp_2+stride[6],cmp_2+stride[7]),a[3],LOADU_256_TO_512(cmp_3+stride[6],cmp_3+stride[7])
		);
		for(remove_cv_t<decltype(32)> i(8),j(4);i!=32;i+=4,j+=2)
		{
			t512[0]=_mm512_add_epi32(t512[0]
				,MULTIPLE_SAD_ADD_0_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[j],LOADU_256_TO_512(cmp_0+stride[i],cmp_0+stride[i+1]),a[j],LOADU_256_TO_512(cmp_1+stride[i],cmp_1+stride[i+1])
					,a[j+1],LOADU_256_TO_512(cmp_0+stride[i+2],cmp_0+stride[i+3]),a[j+1],LOADU_256_TO_512(cmp_1+stride[i+2],cmp_1+stride[i+3])
				)
			);
			t512[1]=_mm512_add_epi32(t512[1]
				,MULTIPLE_SAD_ADD_0_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[j],LOADU_256_TO_512(cmp_2+stride[i],cmp_2+stride[i+1]),a[j],LOADU_256_TO_512(cmp_3+stride[i],cmp_3+stride[i+1])
					,a[j+1],LOADU_256_TO_512(cmp_2+stride[i+2],cmp_2+stride[i+3]),a[j+1],LOADU_256_TO_512(cmp_3+stride[i+2],cmp_3+stride[i+3])
				)
			);
		}
		t512[0]=_mm512_add_epi32(_mm512_unpacklo_epi64(t512[0],t512[1]),_mm512_unpackhi_epi64(t512[0],t512[1]));
		*des=EXTRACT512_TO_128X4_ADD32(t512[0]);
	}

	void sad_512x16(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const uint8_t *cmp_4,const uint8_t *cmp_5
		,const uint8_t *cmp_6,const uint8_t *cmp_7
		,const int32_t *stride,__m256i *des
	)
	{
		__m512i t512[2];
		t512[0]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_256_TO_512(cmp_0,cmp_4),a[0],LOADU_256_TO_512(cmp_1,cmp_5)
			,a[1],LOADU_256_TO_512(cmp_0+stride[1],cmp_4+stride[1]),a[1],LOADU_256_TO_512(cmp_1+stride[1],cmp_5+stride[1])
			,a[2],LOADU_256_TO_512(cmp_0+stride[2],cmp_4+stride[2]),a[2],LOADU_256_TO_512(cmp_1+stride[2],cmp_5+stride[2])
			,a[3],LOADU_256_TO_512(cmp_0+stride[3],cmp_4+stride[3]),a[3],LOADU_256_TO_512(cmp_1+stride[3],cmp_5+stride[3])
		);
		t512[1]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],LOADU_256_TO_512(cmp_2,cmp_6),a[0],LOADU_256_TO_512(cmp_3,cmp_7)
			,a[1],LOADU_256_TO_512(cmp_2+stride[1],cmp_6+stride[1]),a[1],LOADU_256_TO_512(cmp_3+stride[1],cmp_7+stride[1])
			,a[2],LOADU_256_TO_512(cmp_2+stride[2],cmp_6+stride[2]),a[2],LOADU_256_TO_512(cmp_3+stride[2],cmp_7+stride[2])
			,a[3],LOADU_256_TO_512(cmp_2+stride[3],cmp_6+stride[3]),a[3],LOADU_256_TO_512(cmp_3+stride[3],cmp_7+stride[3])
		);
		for(remove_cv_t<decltype(32)> i(4);i!=32;i+=4)
		{
			t512[0]=_mm512_add_epi32(t512[0]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],LOADU_256_TO_512(cmp_0+stride[i],cmp_4+stride[i]),a[i],LOADU_256_TO_512(cmp_1+stride[i],cmp_5+stride[i])
					,a[i+1],LOADU_256_TO_512(cmp_0+stride[i+1],cmp_4+stride[i+1]),a[i+1],LOADU_256_TO_512(cmp_1+stride[i+1],cmp_5+stride[i+1])
					,a[i+2],LOADU_256_TO_512(cmp_0+stride[i+2],cmp_4+stride[i+2]),a[i+2],LOADU_256_TO_512(cmp_1+stride[i+2],cmp_5+stride[i+2])
					,a[i+3],LOADU_256_TO_512(cmp_0+stride[i+3],cmp_4+stride[i+3]),a[i+3],LOADU_256_TO_512(cmp_1+stride[i+3],cmp_5+stride[i+3])
				)
			);
			t512[1]=_mm512_add_epi32(t512[1]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],LOADU_256_TO_512(cmp_2+stride[i],cmp_6+stride[i]),a[i],LOADU_256_TO_512(cmp_3+stride[i],cmp_7+stride[i])
					,a[i+1],LOADU_256_TO_512(cmp_2+stride[i+1],cmp_6+stride[i+1]),a[i+1],LOADU_256_TO_512(cmp_3+stride[i+1],cmp_7+stride[i+1])
					,a[i+2],LOADU_256_TO_512(cmp_2+stride[i+2],cmp_6+stride[i+2]),a[i+2],LOADU_256_TO_512(cmp_3+stride[i+2],cmp_7+stride[i+2])
					,a[i+3],LOADU_256_TO_512(cmp_2+stride[i+3],cmp_6+stride[i+3]),a[i+3],LOADU_256_TO_512(cmp_3+stride[i+3],cmp_7+stride[i+3])
				)
			);
		}
#if 1
		t512[0]=_mm512_add_epi32(_mm512_unpacklo_epi64(t512[0],t512[1]),_mm512_unpackhi_epi64(t512[0],t512[1]));
#else
		//slower
		t512[0]=_mm512_unpacklo_epi64(_mm512_add_epi32(t512[0],_mm512_shuffle_epi32(t512[0],_MM_PERM_BADC)),_mm512_add_epi32(t512[1],_mm512_shuffle_epi32(t512[1],_MM_PERM_BADC)));
#endif
		*des=_mm256_add_epi32(
			_mm512_castsi512_si256(_mm512_permutexvar_epi64(m512_epi64_5_4_1_0,t512[0]))
			,_mm512_castsi512_si256(_mm512_permutexvar_epi64(m512_epi64_7_6_3_2,t512[0]))
		);
	}

	int32_t sad_512x64(const __m512i *a,const uint8_t *src,const int32_t *stride)
	{
		__m512i t512(
			_mm512_add_epi32(
				MULTIPLE_SAD_ADD_3(
					MULTIPLE_SAD_BY_ADD
					,a[0],_mm512_loadu_si512(src)
					,a[1],_mm512_loadu_si512(src+stride[1])
					,a[2],_mm512_loadu_si512(src+stride[2])
					,a[3],_mm512_loadu_si512(src+stride[3])
					,a[4],_mm512_loadu_si512(src+stride[4])
					,a[5],_mm512_loadu_si512(src+stride[5])
					,a[6],_mm512_loadu_si512(src+stride[6])
					,a[7],_mm512_loadu_si512(src+stride[7])
					,a[8],_mm512_loadu_si512(src+stride[8])
					,a[9],_mm512_loadu_si512(src+stride[9])
					,a[10],_mm512_loadu_si512(src+stride[10])
					,a[11],_mm512_loadu_si512(src+stride[11])
					,a[12],_mm512_loadu_si512(src+stride[12])
					,a[13],_mm512_loadu_si512(src+stride[13])
					,a[14],_mm512_loadu_si512(src+stride[14])
					,a[15],_mm512_loadu_si512(src+stride[15])
					,a[16],_mm512_loadu_si512(src+stride[16])
					,a[17],_mm512_loadu_si512(src+stride[17])
					,a[18],_mm512_loadu_si512(src+stride[18])
					,a[19],_mm512_loadu_si512(src+stride[19])
					,a[20],_mm512_loadu_si512(src+stride[20])
					,a[21],_mm512_loadu_si512(src+stride[21])
					,a[22],_mm512_loadu_si512(src+stride[22])
					,a[23],_mm512_loadu_si512(src+stride[23])
					,a[24],_mm512_loadu_si512(src+stride[24])
					,a[25],_mm512_loadu_si512(src+stride[25])
					,a[26],_mm512_loadu_si512(src+stride[26])
					,a[27],_mm512_loadu_si512(src+stride[27])
					,a[28],_mm512_loadu_si512(src+stride[28])
					,a[29],_mm512_loadu_si512(src+stride[29])
					,a[30],_mm512_loadu_si512(src+stride[30])
					,a[31],_mm512_loadu_si512(src+stride[31])
				)
				,MULTIPLE_SAD_ADD_3(
					MULTIPLE_SAD_BY_ADD
					,a[32],_mm512_loadu_si512(src+stride[32])
					,a[33],_mm512_loadu_si512(src+stride[33])
					,a[34],_mm512_loadu_si512(src+stride[34])
					,a[35],_mm512_loadu_si512(src+stride[35])
					,a[36],_mm512_loadu_si512(src+stride[36])
					,a[37],_mm512_loadu_si512(src+stride[37])
					,a[38],_mm512_loadu_si512(src+stride[38])
					,a[39],_mm512_loadu_si512(src+stride[39])
					,a[40],_mm512_loadu_si512(src+stride[40])
					,a[41],_mm512_loadu_si512(src+stride[41])
					,a[42],_mm512_loadu_si512(src+stride[42])
					,a[43],_mm512_loadu_si512(src+stride[43])
					,a[44],_mm512_loadu_si512(src+stride[44])
					,a[45],_mm512_loadu_si512(src+stride[45])
					,a[46],_mm512_loadu_si512(src+stride[46])
					,a[47],_mm512_loadu_si512(src+stride[47])
					,a[48],_mm512_loadu_si512(src+stride[48])
					,a[49],_mm512_loadu_si512(src+stride[49])
					,a[50],_mm512_loadu_si512(src+stride[50])
					,a[51],_mm512_loadu_si512(src+stride[51])
					,a[52],_mm512_loadu_si512(src+stride[52])
					,a[53],_mm512_loadu_si512(src+stride[53])
					,a[54],_mm512_loadu_si512(src+stride[54])
					,a[55],_mm512_loadu_si512(src+stride[55])
					,a[56],_mm512_loadu_si512(src+stride[56])
					,a[57],_mm512_loadu_si512(src+stride[57])
					,a[58],_mm512_loadu_si512(src+stride[58])
					,a[59],_mm512_loadu_si512(src+stride[59])
					,a[60],_mm512_loadu_si512(src+stride[60])
					,a[61],_mm512_loadu_si512(src+stride[61])
					,a[62],_mm512_loadu_si512(src+stride[62])
					,a[63],_mm512_loadu_si512(src+stride[63])
				)
			)
		);
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		return ADD_128_0_2(t128);
	}

	void sad_512x64(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const int32_t *stride,__m128i *des
	)
	{
		__m512i t512(MULTIPLE_SAD_ADD_3(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_0),a[0],_mm512_loadu_si512(cmp_1)
			,a[1],_mm512_loadu_si512(cmp_0+stride[1]),a[1],_mm512_loadu_si512(cmp_1+stride[1])
			,a[2],_mm512_loadu_si512(cmp_0+stride[2]),a[2],_mm512_loadu_si512(cmp_1+stride[2])
			,a[3],_mm512_loadu_si512(cmp_0+stride[3]),a[3],_mm512_loadu_si512(cmp_1+stride[3])
			,a[4],_mm512_loadu_si512(cmp_0+stride[4]),a[4],_mm512_loadu_si512(cmp_1+stride[4])
			,a[5],_mm512_loadu_si512(cmp_0+stride[5]),a[5],_mm512_loadu_si512(cmp_1+stride[5])
			,a[6],_mm512_loadu_si512(cmp_0+stride[6]),a[6],_mm512_loadu_si512(cmp_1+stride[6])
			,a[7],_mm512_loadu_si512(cmp_0+stride[7]),a[7],_mm512_loadu_si512(cmp_1+stride[7])
			,a[8],_mm512_loadu_si512(cmp_0+stride[8]),a[8],_mm512_loadu_si512(cmp_1+stride[8])
			,a[9],_mm512_loadu_si512(cmp_0+stride[9]),a[9],_mm512_loadu_si512(cmp_1+stride[9])
			,a[10],_mm512_loadu_si512(cmp_0+stride[10]),a[10],_mm512_loadu_si512(cmp_1+stride[10])
			,a[11],_mm512_loadu_si512(cmp_0+stride[11]),a[11],_mm512_loadu_si512(cmp_1+stride[11])
			,a[12],_mm512_loadu_si512(cmp_0+stride[12]),a[12],_mm512_loadu_si512(cmp_1+stride[12])
			,a[13],_mm512_loadu_si512(cmp_0+stride[13]),a[13],_mm512_loadu_si512(cmp_1+stride[13])
			,a[14],_mm512_loadu_si512(cmp_0+stride[14]),a[14],_mm512_loadu_si512(cmp_1+stride[14])
			,a[15],_mm512_loadu_si512(cmp_0+stride[15]),a[15],_mm512_loadu_si512(cmp_1+stride[15])
		));
		for(remove_cv_t<decltype(64)> i(16);i!=64;i+=16)
			t512=_mm512_add_epi32(t512
				,MULTIPLE_SAD_ADD_3(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_0+stride[i]),a[i],_mm512_loadu_si512(cmp_1+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_0+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_1+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_0+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_1+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_0+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_1+stride[i+3])
					,a[i+4],_mm512_loadu_si512(cmp_0+stride[i+4]),a[i+4],_mm512_loadu_si512(cmp_1+stride[i+4])
					,a[i+5],_mm512_loadu_si512(cmp_0+stride[i+5]),a[i+5],_mm512_loadu_si512(cmp_1+stride[i+5])
					,a[i+6],_mm512_loadu_si512(cmp_0+stride[i+6]),a[i+6],_mm512_loadu_si512(cmp_1+stride[i+6])
					,a[i+7],_mm512_loadu_si512(cmp_0+stride[i+7]),a[i+7],_mm512_loadu_si512(cmp_1+stride[i+7])
					,a[i+8],_mm512_loadu_si512(cmp_0+stride[i+8]),a[i+8],_mm512_loadu_si512(cmp_1+stride[i+8])
					,a[i+9],_mm512_loadu_si512(cmp_0+stride[i+9]),a[i+9],_mm512_loadu_si512(cmp_1+stride[i+9])
					,a[i+10],_mm512_loadu_si512(cmp_0+stride[i+10]),a[i+10],_mm512_loadu_si512(cmp_1+stride[i+10])
					,a[i+11],_mm512_loadu_si512(cmp_0+stride[i+11]),a[i+11],_mm512_loadu_si512(cmp_1+stride[i+11])
					,a[i+12],_mm512_loadu_si512(cmp_0+stride[i+12]),a[i+12],_mm512_loadu_si512(cmp_1+stride[i+12])
					,a[i+13],_mm512_loadu_si512(cmp_0+stride[i+13]),a[i+13],_mm512_loadu_si512(cmp_1+stride[i+13])
					,a[i+14],_mm512_loadu_si512(cmp_0+stride[i+14]),a[i+14],_mm512_loadu_si512(cmp_1+stride[i+14])
					,a[i+15],_mm512_loadu_si512(cmp_0+stride[i+15]),a[i+15],_mm512_loadu_si512(cmp_1+stride[i+15])
				)
			);
		__m128i t128(EXTRACT512_TO_128X4_ADD32(t512));
		*des=_mm_add_epi32(t128,_mm_shuffle_epi32(t128,_MM_PERM_BADC));
	}

	void sad_512x64(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const int32_t *stride,__m128i *des
	)
	{
		__m512i t512[3];
		t512[0]=MULTIPLE_SAD_ADD_2(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_0),a[0],_mm512_loadu_si512(cmp_1)
			,a[1],_mm512_loadu_si512(cmp_0+stride[1]),a[1],_mm512_loadu_si512(cmp_1+stride[1])
			,a[2],_mm512_loadu_si512(cmp_0+stride[2]),a[2],_mm512_loadu_si512(cmp_1+stride[2])
			,a[3],_mm512_loadu_si512(cmp_0+stride[3]),a[3],_mm512_loadu_si512(cmp_1+stride[3])
			,a[4],_mm512_loadu_si512(cmp_0+stride[4]),a[4],_mm512_loadu_si512(cmp_1+stride[4])
			,a[5],_mm512_loadu_si512(cmp_0+stride[5]),a[5],_mm512_loadu_si512(cmp_1+stride[5])
			,a[6],_mm512_loadu_si512(cmp_0+stride[6]),a[6],_mm512_loadu_si512(cmp_1+stride[6])
			,a[7],_mm512_loadu_si512(cmp_0+stride[7]),a[7],_mm512_loadu_si512(cmp_1+stride[7])
		);
		t512[1]=MULTIPLE_SAD_ADD_2(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_2),a[0],_mm512_loadu_si512(cmp_3)
			,a[1],_mm512_loadu_si512(cmp_2+stride[1]),a[1],_mm512_loadu_si512(cmp_3+stride[1])
			,a[2],_mm512_loadu_si512(cmp_2+stride[2]),a[2],_mm512_loadu_si512(cmp_3+stride[2])
			,a[3],_mm512_loadu_si512(cmp_2+stride[3]),a[3],_mm512_loadu_si512(cmp_3+stride[3])
			,a[4],_mm512_loadu_si512(cmp_2+stride[4]),a[4],_mm512_loadu_si512(cmp_3+stride[4])
			,a[5],_mm512_loadu_si512(cmp_2+stride[5]),a[5],_mm512_loadu_si512(cmp_3+stride[5])
			,a[6],_mm512_loadu_si512(cmp_2+stride[6]),a[6],_mm512_loadu_si512(cmp_3+stride[6])
			,a[7],_mm512_loadu_si512(cmp_2+stride[7]),a[7],_mm512_loadu_si512(cmp_3+stride[7])
		);
		for(remove_cv_t<decltype(64)> i(8);i!=64;i+=8)
		{
			t512[0]=_mm512_add_epi32(t512[0]
				,MULTIPLE_SAD_ADD_2(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_0+stride[i]),a[i],_mm512_loadu_si512(cmp_1+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_0+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_1+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_0+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_1+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_0+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_1+stride[i+3])
					,a[i+4],_mm512_loadu_si512(cmp_0+stride[i+4]),a[i+4],_mm512_loadu_si512(cmp_1+stride[i+4])
					,a[i+5],_mm512_loadu_si512(cmp_0+stride[i+5]),a[i+5],_mm512_loadu_si512(cmp_1+stride[i+5])
					,a[i+6],_mm512_loadu_si512(cmp_0+stride[i+6]),a[i+6],_mm512_loadu_si512(cmp_1+stride[i+6])
					,a[i+7],_mm512_loadu_si512(cmp_0+stride[i+7]),a[i+7],_mm512_loadu_si512(cmp_1+stride[i+7])
				)
			);
			t512[1]=_mm512_add_epi32(t512[1]
				,MULTIPLE_SAD_ADD_2(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_2+stride[i]),a[i],_mm512_loadu_si512(cmp_3+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_2+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_3+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_2+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_3+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_2+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_3+stride[i+3])
					,a[i+4],_mm512_loadu_si512(cmp_2+stride[i+4]),a[i+4],_mm512_loadu_si512(cmp_3+stride[i+4])
					,a[i+5],_mm512_loadu_si512(cmp_2+stride[i+5]),a[i+5],_mm512_loadu_si512(cmp_3+stride[i+5])
					,a[i+6],_mm512_loadu_si512(cmp_2+stride[i+6]),a[i+6],_mm512_loadu_si512(cmp_3+stride[i+6])
					,a[i+7],_mm512_loadu_si512(cmp_2+stride[i+7]),a[i+7],_mm512_loadu_si512(cmp_3+stride[i+7])
				)
			);
		}
		t512[2]=_mm512_add_epi32(_mm512_unpacklo_epi64(t512[0],t512[1]),_mm512_unpackhi_epi64(t512[0],t512[1]));
		*des=EXTRACT512_TO_128X4_ADD32(t512[2]);
	}
	
	void sad_512x64(const __m512i *a
		,const uint8_t *cmp_0,const uint8_t *cmp_1
		,const uint8_t *cmp_2,const uint8_t *cmp_3
		,const uint8_t *cmp_4,const uint8_t *cmp_5
		,const uint8_t *cmp_6,const uint8_t *cmp_7
		,const int32_t *stride,__m256i *des
	)
	{
		__m512i t512[6];
		t512[0]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_0),a[0],_mm512_loadu_si512(cmp_1)
			,a[1],_mm512_loadu_si512(cmp_0+stride[1]),a[1],_mm512_loadu_si512(cmp_1+stride[1])
			,a[2],_mm512_loadu_si512(cmp_0+stride[2]),a[2],_mm512_loadu_si512(cmp_1+stride[2])
			,a[3],_mm512_loadu_si512(cmp_0+stride[3]),a[3],_mm512_loadu_si512(cmp_1+stride[3])
		);
		t512[1]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_2),a[0],_mm512_loadu_si512(cmp_3)
			,a[1],_mm512_loadu_si512(cmp_2+stride[1]),a[1],_mm512_loadu_si512(cmp_3+stride[1])
			,a[2],_mm512_loadu_si512(cmp_2+stride[2]),a[2],_mm512_loadu_si512(cmp_3+stride[2])
			,a[3],_mm512_loadu_si512(cmp_2+stride[3]),a[3],_mm512_loadu_si512(cmp_3+stride[3])
		);
		t512[2]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_4),a[0],_mm512_loadu_si512(cmp_5)
			,a[1],_mm512_loadu_si512(cmp_4+stride[1]),a[1],_mm512_loadu_si512(cmp_5+stride[1])
			,a[2],_mm512_loadu_si512(cmp_4+stride[2]),a[2],_mm512_loadu_si512(cmp_5+stride[2])
			,a[3],_mm512_loadu_si512(cmp_4+stride[3]),a[3],_mm512_loadu_si512(cmp_5+stride[3])
		);
		t512[3]=MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
			MULTIPLE_SAD
			,a[0],_mm512_loadu_si512(cmp_6),a[0],_mm512_loadu_si512(cmp_7)
			,a[1],_mm512_loadu_si512(cmp_6+stride[1]),a[1],_mm512_loadu_si512(cmp_7+stride[1])
			,a[2],_mm512_loadu_si512(cmp_6+stride[2]),a[2],_mm512_loadu_si512(cmp_7+stride[2])
			,a[3],_mm512_loadu_si512(cmp_6+stride[3]),a[3],_mm512_loadu_si512(cmp_7+stride[3])
		);
		for(remove_cv_t<decltype(64)> i(4);i!=64;i+=4)
		{
			t512[0]=_mm512_add_epi32(t512[0]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_0+stride[i]),a[i],_mm512_loadu_si512(cmp_1+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_0+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_1+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_0+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_1+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_0+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_1+stride[i+3])
				)
			);
			t512[1]=_mm512_add_epi32(t512[1]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_2+stride[i]),a[i],_mm512_loadu_si512(cmp_3+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_2+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_3+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_2+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_3+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_2+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_3+stride[i+3])
				)
			);
			t512[2]=_mm512_add_epi32(t512[2]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_4+stride[i]),a[i],_mm512_loadu_si512(cmp_5+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_4+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_5+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_4+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_5+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_4+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_5+stride[i+3])
				)
			);
			t512[3]=_mm512_add_epi32(t512[3]
				,MULTIPLE_SAD_ADD_1_ADD_THEN_SHUFFLE(
					MULTIPLE_SAD
					,a[i],_mm512_loadu_si512(cmp_6+stride[i]),a[i],_mm512_loadu_si512(cmp_7+stride[i])
					,a[i+1],_mm512_loadu_si512(cmp_6+stride[i+1]),a[i+1],_mm512_loadu_si512(cmp_7+stride[i+1])
					,a[i+2],_mm512_loadu_si512(cmp_6+stride[i+2]),a[i+2],_mm512_loadu_si512(cmp_7+stride[i+2])
					,a[i+3],_mm512_loadu_si512(cmp_6+stride[i+3]),a[i+3],_mm512_loadu_si512(cmp_7+stride[i+3])
				)
			);
		}
		t512[4]=_mm512_add_epi32(_mm512_unpacklo_epi64(t512[0],t512[1]),_mm512_unpackhi_epi64(t512[0],t512[1]));
		t512[5]=_mm512_add_epi32(_mm512_unpacklo_epi64(t512[2],t512[3]),_mm512_unpackhi_epi64(t512[2],t512[3]));
#if 1	//they have almost same performance
		__m256i t256[2];
		t256[0]=_mm256_add_epi32(_mm512_castsi512_si256(t512[4]),_mm512_extracti64x4_epi64(t512[4],1));
		t256[1]=_mm256_add_epi32(_mm512_castsi512_si256(t512[5]),_mm512_extracti64x4_epi64(t512[5],1));
		*des=_mm256_add_epi32(
			_mm256_shuffle_i64x2(t256[0],t256[1],0b0001)
			,_mm256_blend_epi32(t256[0],t256[1],0b11110000)
		);
#else
		__m512i t512_6(_mm512_add_epi32(
			_mm512_permutex2var_epi64(t512[4],m512_epi64_13_12_5_4_9_8_1_0,t512[5])
			,_mm512_permutex2var_epi64(t512[4],m512_epi64_15_14_7_6_11_10_3_2,t512[5])
		));
		*des=_mm256_add_epi32(_mm512_castsi512_si256(t512_6),_mm512_extracti64x4_epi64(t512_6,1));
#endif
	}
}