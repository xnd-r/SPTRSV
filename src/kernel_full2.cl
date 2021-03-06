#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define WARP_SIZE 64

#define SUBSTITUTION_BACKWARD 1

#define OPT_WARP_NNZ   1
#define OPT_WARP_RHS   2
#define OPT_WARP_AUTO  3

inline
void atom_add_d_fp64(volatile __global double* val,
    double delta)
{
    union { double f; ulong i; } old;
    union { double f; ulong i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    } while (atom_cmpxchg((volatile __global ulong*)val, old.i, new.i) != old.i);
}

inline
void atom_add_s_fp64(volatile __local double* val,
    double delta)
{
    union { double f; ulong i; } old;
    union { double f; ulong i; } new;
    do
    {
        old.f = *val;
        new.f = old.f + delta;
    } while (atom_cmpxchg((volatile __local ulong*)val, old.i, new.i) != old.i);
}

__kernel
void sptrsv_syncfree_opencl_analyser(__global const int* d_cscRowIdx,
    const int                m,
    const int                nnz,
    __global int* d_graphInDegree)
{
    const int global_id = get_global_id(0);
    if (global_id < nnz)
    {
        atomic_fetch_add_explicit((atomic_int*)&d_graphInDegree[d_cscRowIdx[global_id]], 1,
            memory_order_acq_rel, memory_scope_device);
    }
}

__kernel
void sptrsv_syncfree_opencl_executor(__global const int* d_cscColPtr,
    __global const int* d_cscRowIdx,
    __global const VALUE_TYPE* d_cscVal,
    __global volatile int* d_graphInDegree,
    __global volatile VALUE_TYPE* d_left_sum,
    const int                      m,
    __global const VALUE_TYPE* d_b,
    __global VALUE_TYPE* d_x,
    const int                      warp_per_block)
{
    const int global_id = get_global_id(0);
    const int local_id = get_local_id(0);
    int global_x_id = global_id / WARP_SIZE;
    if (global_x_id >= m) return;

    // substitution is forward or backward
    global_x_id = m - 1 - global_x_id;

    // Initialize
    const int local_warp_id = local_id / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & local_id;
    int starting_x = (global_id / (warp_per_block * WARP_SIZE)) * warp_per_block;
    starting_x = m - 1 - starting_x;

    // Prefetch
    const int pos = d_cscColPtr[global_x_id + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];


    // Consumer
    int loads, loadd;
    do {
        // busy-wait
    }
    while (1 !=
           (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],
                                         memory_order_acquire, memory_scope_device)) );

    VALUE_TYPE xi = d_left_sum[global_x_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = d_cscColPtr[global_x_id];
    const int stop_ptr = d_cscColPtr[global_x_id + 1] - 1;
    for (int j = start_ptr + lane_id; j < stop_ptr; j += WARP_SIZE) {
        const int rowIdx = d_cscRowIdx[j];
        atom_add_d_fp64(&d_left_sum[rowIdx], xi * d_cscVal[j]);
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
            memory_order_acquire, memory_scope_device);
    }

    // Finish
    if (!lane_id) d_x[global_x_id] = xi;
}
