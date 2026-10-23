// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

using B2 = __bf16 __attribute__((ext_vector_type(2)));
using F2 = float __attribute__((ext_vector_type(2)));
constexpr int Begin = 0x3980;
constexpr int Entries = 0x4180-Begin;

template <class T>
__device__ __forceinline__ T lookup(uint16_t code,const T* hot,const T* full) {
  const int magnitude=code&0x7fff;
  if (magnitude>=Begin && magnitude<Begin+Entries)
    return hot[magnitude-Begin+((code&0x8000)?Entries:0)];
  return full[code];
}

template <bool Backward,int Pairs>
__global__ void Gate(const uint32_t* __restrict__ x,
                     const uint32_t* linear,
                     const uint32_t* __restrict__ grad,
                     const void* __restrict__ table,
                     uint32_t* out0,
                     uint32_t* __restrict__ out1,uint32_t count) {
  using Entry=typename std::conditional<Backward,uint64_t,uint32_t>::type;
  __shared__ Entry hot[2*Entries];
  const auto* full=static_cast<const Entry*>(table);
  for (int i=threadIdx.x;i<2*Entries;i+=256)
    hot[i]=full[Begin+(i%Entries)+(i>=Entries?0x8000:0)];
  __syncthreads();
  const uint32_t stride=gridDim.x*256*Pairs;
  for (uint32_t start=blockIdx.x*256*Pairs+threadIdx.x;start<count;start+=stride) {
    #pragma unroll
    for (int j=0;j<Pairs;++j) {
      uint32_t i=start+uint32_t(j)*256;
      if (i<count) {
        uint32_t codes=x[i];
        Entry lo=lookup(uint16_t(codes),hot,full);
        Entry hi=lookup(uint16_t(codes>>16),hot,full);
        F2 value={__builtin_bit_cast(float,uint32_t(lo)),__builtin_bit_cast(float,uint32_t(hi))};
        F2 up=__builtin_convertvector(__builtin_bit_cast(B2,linear[i]),F2);
        if constexpr (Backward) {
          F2 derivative={__builtin_bit_cast(float,uint32_t(lo>>32)),__builtin_bit_cast(float,uint32_t(hi>>32))};
          F2 g=__builtin_convertvector(__builtin_bit_cast(B2,grad[i]),F2);
          F2 product=g*up;
          B2 dx=__builtin_convertvector(product*derivative,B2);
          B2 dl=__builtin_convertvector(g*value,B2);
          out0[i]=__builtin_bit_cast(uint32_t,dx);
          out1[i]=__builtin_bit_cast(uint32_t,dl);
        } else {
          B2 y=__builtin_convertvector(value*up,B2);
          out0[i]=__builtin_bit_cast(uint32_t,y);
        }
      }
    }
  }
}

template<bool B,int P>
void launch(int grid,hipStream_t stream,const void* x,const void* linear,
            const void* grad,const void* table,void* out0,void* out1,uint32_t n) {
  hipLaunchKernelGGL((Gate<B,P>),dim3(grid),dim3(256),0,stream,
    static_cast<const uint32_t*>(x),static_cast<const uint32_t*>(linear),
    static_cast<const uint32_t*>(grad),table,static_cast<uint32_t*>(out0),
    static_cast<uint32_t*>(out1),n/2);
}

extern "C" int openpi_torch_gate(int backward,int pairs,int grid,void* stream,
    const void* x,const void* linear,const void* grad,const void* table,
    void* out0,void* out1,uint32_t n) {
  if (n%2 || n>2147483647 || grid<1) return -1;
  hipStream_t s=static_cast<hipStream_t>(stream);
  if (backward) {
    if (pairs==1) launch<true,1>(grid,s,x,linear,grad,table,out0,out1,n);
    else if (pairs==2) launch<true,2>(grid,s,x,linear,grad,table,out0,out1,n);
    else if (pairs==4) launch<true,4>(grid,s,x,linear,grad,table,out0,out1,n);
    else return -2;
  } else {
    if (pairs==1) launch<false,1>(grid,s,x,linear,grad,table,out0,out1,n);
    else if (pairs==2) launch<false,2>(grid,s,x,linear,grad,table,out0,out1,n);
    else if (pairs==4) launch<false,4>(grid,s,x,linear,grad,table,out0,out1,n);
    else return -2;
  }
  return int(hipGetLastError());
}
