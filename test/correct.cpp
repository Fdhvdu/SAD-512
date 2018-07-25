#include<atomic>
#include<chrono>
#include<cstdint>
#include<fstream>
#include<iostream>
#include<limits>
#include<new>
#include<numeric>
#include<thread>
#include<type_traits>
#include<vector>
#include"SAD-512/header/intel_sad.hpp"
using namespace std;

namespace
{
	constexpr auto cnt(1000);
	constexpr auto img_size(64);	//8, 16, 32, 64
	constexpr auto batch(8);	//1, 2, 4, 8
	constexpr auto width(cnt+63);
	constexpr auto height(cnt+63);
	constexpr auto stride(width);
	constexpr int32_t min_sad[4]{3184,17683,79079,328899};
#if 1
	const auto thr_cnt(1);
#else
	const auto thr_cnt(thread::hardware_concurrency());
#endif

	void (*loadu)(const uint8_t *,const int32_t *,__m512i *);
	int32_t (*sadx1)(const __m512i *,const uint8_t *,const int32_t *);
	void (*sadx2)(const __m512i *,const uint8_t *,const uint8_t *,const int32_t *,__m128i *);
	void (*sadx4)(const __m512i *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const int32_t *,__m128i *);
	void (*sadx8)(const __m512i *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const int32_t *,__m256i *);

	atomic<unsigned int> iamready(0);
	atomic<bool> spin(true);
	int64_t *elapsed_time(static_cast<int64_t *>(operator new[](64*thr_cnt,align_val_t{64})));	//prevent false sharing
	static_assert(sizeof(chrono::nanoseconds::rep)<=64);

	vector<uint8_t> pixel(stride*height);
	alignas(64) int32_t mul_stride[64];
	__m512i t512[64];

	void init_func_pointer();
	void load_pixel(const char *path);
	void load_t512(const char *path);
	void task(const unsigned int id);
}

int main()
{
	init_func_pointer();
	new (elapsed_time) int64_t[thr_cnt];
	nIntel::mul_stridex512x4(stride,reinterpret_cast<__m512i*>(mul_stride));
	load_pixel("pixel");
	load_t512("cmp");
	{
		vector<thread> thr;
		{
			thr.reserve(thr_cnt);
			for(remove_cv_t<decltype(thr_cnt)> i(0);i!=thr_cnt;++i)
				thr.emplace_back(task,i);
			while(iamready!=thr_cnt)
				;
			spin=false;
		}
		for(auto &val:thr)
			val.join();
	}
	cout<<static_cast<long double>(accumulate(elapsed_time,elapsed_time+thr_cnt,static_cast<chrono::nanoseconds::rep>(0)))/(cnt*((cnt+batch-1)/batch)*thr_cnt)<<endl;
	delete []elapsed_time;
}

namespace
{
	void init_func_pointer()
	{
		using namespace nIntel;
		if constexpr(img_size==8)
		{
			loadu=loadu_64x8;
			sadx1=sad_512x1;
			sadx2=sad_512x1;
			sadx4=sad_512x1;
			sadx8=sad_512x1;
		}
		else
			if constexpr(img_size==16)
			{
				loadu=loadu_128x16;
				sadx1=sad_512x4;
				sadx2=sad_512x4;
				sadx4=sad_512x4;
				sadx8=sad_512x4;
			}
			else
				if constexpr(img_size==32)
				{
					loadu=loadu_256x32;
					sadx1=sad_512x16;
					sadx2=sad_512x16;
					sadx4=sad_512x16;
					sadx8=sad_512x16;
				}
				else
					if constexpr(img_size==64)
					{
						loadu=loadu_512x64;
						sadx1=sad_512x64;
						sadx2=sad_512x64;
						sadx4=sad_512x64;
						sadx8=sad_512x64;
					}
	}
	
	void load_pixel(const char *path)
	{
		ifstream ifs(path);
		ifs.read(reinterpret_cast<char*>(pixel.data()),pixel.size());
	}

	void load_t512(const char *path)
	{
		ifstream ifs(path);
		char img[size(t512)*sizeof(__m512i)];
		ifs.read(img,size(img));
		alignas(64) int32_t local_stride[64];
		nIntel::mul_stridex512x4(sizeof(__m512i),reinterpret_cast<__m512i*>(local_stride));
		loadu(reinterpret_cast<uint8_t*>(img),local_stride,t512);
	}

	void task(const unsigned int id)
	{
		using namespace nIntel;
		alignas(32) int32_t result[8];
		int32_t min(numeric_limits<int32_t>::max());
		++iamready;
		while(spin)
			;
		const auto timer_begin(chrono::steady_clock::now());
		for(volatile remove_cv_t<decltype(height)> y(0);y<cnt;++y)
			for(volatile remove_cv_t<decltype(width)> x(0);x<cnt;x+=batch)
			{
				if constexpr(batch==1)
					result[0]=sadx1(t512
						,pixel.data()+y*mul_stride[1]+x
						,mul_stride
					);
				else
					if constexpr(batch==2)
						sadx2(t512
							,pixel.data()+y*mul_stride[1]+x
							,pixel.data()+y*mul_stride[1]+x+1
							,mul_stride
							,reinterpret_cast<__m128i*>(result)
						);
					else
						if constexpr(batch==4)
							sadx4(t512
								,pixel.data()+y*mul_stride[1]+x
								,pixel.data()+y*mul_stride[1]+x+1
								,pixel.data()+y*mul_stride[1]+x+2
								,pixel.data()+y*mul_stride[1]+x+3
								,mul_stride
								,reinterpret_cast<__m128i*>(result)
							);
						else
							if constexpr(batch==8)
								sadx8(t512
									,pixel.data()+y*mul_stride[1]+x
									,pixel.data()+y*mul_stride[1]+x+1
									,pixel.data()+y*mul_stride[1]+x+2
									,pixel.data()+y*mul_stride[1]+x+3
									,pixel.data()+y*mul_stride[1]+x+4
									,pixel.data()+y*mul_stride[1]+x+5
									,pixel.data()+y*mul_stride[1]+x+6
									,pixel.data()+y*mul_stride[1]+x+7
									,mul_stride
									,reinterpret_cast<__m256i*>(result)
								);
				for(remove_cv_t<decltype(batch)> i(0);i!=batch;++i)
					if(result[i]<min)
						min=result[i];
			}
		const auto timer_end(chrono::steady_clock::now());
		elapsed_time[id]=chrono::duration_cast<chrono::nanoseconds>(timer_end-timer_begin).count();
		if constexpr(img_size==8)
		{
			if(min!=min_sad[0])
				cerr<<"cannot pass test"<<endl;
		}
		else
			if constexpr(img_size==16)
			{
				if(min!=min_sad[1])
					cerr<<"cannot pass test"<<endl;
			}
			else
				if constexpr(img_size==32)
				{
					if(min!=min_sad[2])
						cerr<<"cannot pass test"<<endl;
				}
				else
					if constexpr(img_size==64)
					{
						if(min!=min_sad[3])
							cerr<<"cannot pass test"<<endl;
					}
	}
}