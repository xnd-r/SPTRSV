#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif
#define WARP_SIZE 64


__kernel
void sptrsv_syncfree_opencl_executor(__global const int            *d_csrRowPtr,
                                     __global const int            *d_csrColIdx,
                                     __global const VALUE_TYPE     *d_csrVal,
                                     __global volatile int         *d_get_value,
                                     const int                      m,
                                     __global const VALUE_TYPE     *d_b,
                                     __global volatile VALUE_TYPE           *d_x)
{
    const int global_id = get_global_id(0);
    if(global_id>=m)
        return;

    int col,j,i;
    VALUE_TYPE xi;
    VALUE_TYPE left_sum=0;
    i=global_id;
    j=d_csrRowPtr[i];

    while(j<d_csrRowPtr[i+1])
    {
        col=d_csrColIdx[j];
        if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
        //while(d_get_value[col]==1)
            //if(d_get_value[col]==1)
        {
            left_sum+=d_csrVal[j]*d_x[col];
            j++;
            col=d_csrColIdx[j];
        }
        if(i==col)
        {
            xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i+1]-1];
            d_x[i] = xi;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            d_get_value[i]=1;
            j++;
        }
    }

}
