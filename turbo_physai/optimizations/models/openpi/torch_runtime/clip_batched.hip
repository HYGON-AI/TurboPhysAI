// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <hip/hip_runtime.h>
#include <cstdint>
struct BF { uint16_t x; };
__device__ __forceinline__ float decode(BF a){return __builtin_bit_cast(float,uint32_t(a.x)<<16);}
__device__ __forceinline__ BF encode(float f){uint32_t u=__builtin_bit_cast(uint32_t,f);return BF{uint16_t((u+0x7fff+((u>>16)&1))>>16)};}
struct Meta { BF* p; uint64_t n; };
struct alignas(16) Pack { BF x[8]; };
__global__ void Scale(const Meta* meta,const float* coefficient){
 const float c=decode(encode(*coefficient));
 // A finite BF16 gradient multiplied by exactly 1 preserves every bit.
 // Non-finite gradients make the native norm/coefficient non-finite or 0,
 // so they use the original multiplication below.
 if(c==1.0f)return;
 const Meta item=meta[blockIdx.y];
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<item.n/8;i+=uint64_t(gridDim.x)*blockDim.x){
  Pack value=reinterpret_cast<Pack*>(item.p)[i];
  #pragma unroll
  for(int k=0;k<8;++k)value.x[k]=encode(decode(value.x[k])*c);
  reinterpret_cast<Pack*>(item.p)[i]=value;
 }
 return;
}
extern "C" int openpi_clip_batch(void* stream,const void* metadata,const void* coefficient,int count,int grid){
 hipLaunchKernelGGL(Scale,dim3(grid,count),dim3(256),0,(hipStream_t)stream,(const Meta*)metadata,(const float*)coefficient);
 return int(hipGetLastError());
}

struct Meta32 { float* p; uint64_t n; };
struct alignas(16) Pack32 { float x[4]; };
__global__ void Scale32(const Meta32* meta,const float* coefficient){
 const float c=*coefficient;
 if(c==1.0f)return;
 const Meta32 item=meta[blockIdx.y];
 for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<item.n/4;i+=uint64_t(gridDim.x)*blockDim.x){
  Pack32 value=reinterpret_cast<Pack32*>(item.p)[i];
  #pragma unroll
  for(int k=0;k<4;++k)value.x[k]*=c;
  reinterpret_cast<Pack32*>(item.p)[i]=value;
 }
 return;
}
extern "C" int openpi_clip32_batch(void* stream,const void* metadata,const void* coefficient,int count,int grid){
 hipLaunchKernelGGL(Scale32,dim3(grid,count),dim3(256),0,(hipStream_t)stream,(const Meta32*)metadata,(const float*)coefficient);
 return int(hipGetLastError());
}
