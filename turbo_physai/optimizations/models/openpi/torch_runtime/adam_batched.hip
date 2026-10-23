// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause
//
// Preserve Torch's explicit FP32 intermediates and software BF16 RNE.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>
struct BF {uint16_t x;};
__device__ __forceinline__ float decode(BF a){return __builtin_bit_cast(float,uint32_t(a.x)<<16);}
__device__ __forceinline__ BF encode(float f){uint32_t u=__builtin_bit_cast(uint32_t,f);return BF{uint16_t((u+0x7fff+((u>>16)&1))>>16)};}
template<int PACK>struct alignas(PACK*2) Packed { BF x[PACK]; };
struct Meta { BF* p; const BF* g; BF* m; BF* v; const float* step; uint64_t n; };
__global__ void Prepare(const Meta* meta,float* scale,int count,double lr,double b1,double b2){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count){double s=double(*meta[i].step);float bc1=float(1.0-::pow(b1,s));float bc2=float(::sqrt(1.0-::pow(b2,s)));scale[3*i]=bc1;scale[3*i+1]=bc2;scale[3*i+2]=float(lr/bc1);}
 return;
}
template<int PACK,int REPEAT>
__global__ void AdamBatch(const Meta* meta,const float* scale,double lr,double b1,double b2,double wd,double eps){
 using Pack=Packed<PACK>;
 const Meta item=meta[blockIdx.y];
 BF* __restrict__ p=item.p;const BF* __restrict__ g=item.g;BF* __restrict__ m=item.m;BF* __restrict__ v=item.v;
 uint64_t n=item.n;
 float bc1=scale[3*blockIdx.y],bc2=scale[3*blockIdx.y+1],ss=scale[3*blockIdx.y+2];
 for(uint64_t start=uint64_t(blockIdx.x)*blockDim.x*REPEAT+threadIdx.x;start<n/PACK;start+=uint64_t(gridDim.x)*blockDim.x*REPEAT){
  #pragma unroll
  for(int j=0;j<REPEAT;++j){
   uint64_t idx=start+uint64_t(j)*blockDim.x;
   if(idx<n/PACK){
    Pack pp=reinterpret_cast<Pack*>(p)[idx],gg=reinterpret_cast<const Pack*>(g)[idx],mm=reinterpret_cast<Pack*>(m)[idx],vv=reinterpret_cast<Pack*>(v)[idx];
    #pragma unroll
    for(int k=0;k<PACK;++k){
     float param=decode(pp.x[k]),grad=decode(gg.x[k]),avg=decode(mm.x[k]),sq=decode(vv.x[k]);
     if(wd!=0)param-=lr*wd*param;
     avg=::fma(b1,double(avg),(1-b1)*double(grad));
     sq=b2*sq+(1-b2)*grad*grad;
     float denom=(::sqrtf(sq)/bc2)+eps;
     param-=ss*avg/denom;
     pp.x[k]=encode(param);mm.x[k]=encode(avg);vv.x[k]=encode(sq);
    }
    reinterpret_cast<Pack*>(p)[idx]=pp;reinterpret_cast<Pack*>(m)[idx]=mm;reinterpret_cast<Pack*>(v)[idx]=vv;
   }
  }
 }
 return;
}

template<int PACK,int R>void launch(const Meta* meta,const float* scale,int count,double lr,double b1,double b2,double wd,double eps,int grid,int threads,hipStream_t stream){
 hipLaunchKernelGGL((AdamBatch<PACK,R>),dim3(grid,count),dim3(threads),0,stream,meta,scale,lr,b1,b2,wd,eps);
 return;
}
extern "C" int openpi_adam_batch(int pack,int repeat,int grid,int threads,void* stream,const void* metadata,void* scales,int count,double lr,double b1,double b2,double wd,double eps){
 const Meta* meta=(const Meta*)metadata;float* scale=(float*)scales;hipStream_t st=(hipStream_t)stream;
 hipLaunchKernelGGL(Prepare,dim3((count+127)/128),dim3(128),0,st,meta,scale,count,lr,b1,b2);
 if(pack==8 && repeat==1)launch<8,1>(meta,scale,count,lr,b1,b2,wd,eps,grid,threads,st);
 else if(pack==16 && repeat==2)launch<16,2>(meta,scale,count,lr,b1,b2,wd,eps,grid,threads,st);
 else return -1;
 return int(hipGetLastError());
}
