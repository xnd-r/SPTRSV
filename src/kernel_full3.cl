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
    int global_id = get_global_id(0);
    if(global_id>=m)
        return;
    global_id = m - 1 - global_id;
    //printf("%d\n", global_id);

    int col,j,i;
    VALUE_TYPE xi;
    VALUE_TYPE left_sum=0;
    i=global_id;
    j=d_csrRowPtr[i+1]-1;
    //printf("I: %d J: %d\n", i, j);
    while(j >= d_csrRowPtr[i])
    {
        col=d_csrColIdx[j];
        //printf("%d\n", d_get_value[col]);
        if(atomic_load_explicit((atomic_int*)&d_get_value[col],memory_order_acquire, memory_scope_device)==1)
        {
            left_sum+=d_csrVal[j]*d_x[col];
            j--;
            col=d_csrColIdx[j];
        }
        if(i==col)
        {
            //printf("%d\n", i);
            xi = (d_b[i] - left_sum) / d_csrVal[d_csrRowPtr[i]];
            d_x[i] = xi;
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            d_get_value[i]=1;
            j--;
        }
    }

}
