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
    __local volatile int* s_graphInDegree,
    __local volatile VALUE_TYPE* s_left_sum,
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

    if (local_id < warp_per_block) { s_graphInDegree[local_id] = 1; s_left_sum[local_id] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Consumer
    int loads, loadd;
    do {
        // busy-wait
    } while ((loads = atomic_load_explicit((atomic_int*)&s_graphInDegree[local_warp_id],
        memory_order_acquire, memory_scope_work_group)) !=
        (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],
            memory_order_acquire, memory_scope_device)));

    VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    // Producer
    const int start_ptr = d_cscColPtr[global_x_id];
    const int stop_ptr = d_cscColPtr[global_x_id + 1] - 1;
    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE) {
        const int j = stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j];
        const bool cond = (rowIdx > starting_x - warp_per_block);
        if (cond) {
            const int pos = starting_x - rowIdx;
            atom_add_s_fp64(&s_left_sum[pos], xi * d_cscVal[j]);
            mem_fence(CLK_LOCAL_MEM_FENCE);
            atomic_fetch_add_explicit((atomic_int*)&s_graphInDegree[pos], 1,
                memory_order_acquire, memory_scope_work_group);
        }
        else {
            atom_add_d_fp64(&d_left_sum[rowIdx], xi * d_cscVal[j]);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
                memory_order_acquire, memory_scope_device);
        }
    }

    // Finish
    if (!lane_id) d_x[global_x_id] = xi;
}

    __kernel
    void sptrsm_syncfree_opencl_executor(__global const int            *d_cscColPtr,
                                         __global const int            *d_cscRowIdx,
                                         __global const VALUE_TYPE     *d_cscVal,
                                         __global volatile int         *d_graphInDegree,
                                         __global volatile VALUE_TYPE  *d_left_sum,
                                         const int                      m,
                                         const int                      substitution,
                                         const int                      rhs,
                                         const int                      opt,
                                         __global const VALUE_TYPE     *d_b,
                                         __global VALUE_TYPE           *d_x,
                                         const int                      warp_per_block)
    {
        const int global_id = get_global_id(0);
        int global_x_id = global_id / WARP_SIZE;
        if (global_x_id >= m) return;

        global_x_id = m - 1 - global_x_id;

        // Initialize
        const int lane_id = (WARP_SIZE - 1) & get_local_id(0);

        // Prefetch
        const int pos = d_cscColPtr[global_x_id+1]-1;
        const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

        // Consumer
        int loadd;
        do {
            // busy-wait
        }
        while (1 != (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],
                                             memory_order_acquire, memory_scope_device)) );

       for (int k = lane_id; k < rhs; k += WARP_SIZE)
       {
           const int pos = global_x_id * rhs + k;
           d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;
       }

       // Producer
       const int start_ptr = d_cscColPtr[global_x_id];
       const int stop_ptr  = d_cscColPtr[global_x_id+1]-1;

       if (opt == OPT_WARP_NNZ)
       {
           for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
           {
               const int j = stop_ptr - 1 - (jj - start_ptr);
               const int rowIdx = d_cscRowIdx[j];
               for (int k = 0; k < rhs; k++)
                   atom_add_d_fp64(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
               mem_fence(CLK_GLOBAL_MEM_FENCE);
               atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
                                          memory_order_acquire, memory_scope_device);
           }
       }
       else if (opt == OPT_WARP_RHS)
       {
           for (int jj = start_ptr; jj < stop_ptr; jj++)
           {
               const int j = stop_ptr - 1 - (jj - start_ptr);
               const int rowIdx = d_cscRowIdx[j];
               for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                   atom_add_d_fp64(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
               mem_fence(CLK_GLOBAL_MEM_FENCE);
               if (!lane_id)
                   atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
                                             memory_order_acquire, memory_scope_device);
           }
       }
       else if (opt == OPT_WARP_AUTO)
       {
           const int len = stop_ptr - start_ptr;

               for (int jj = start_ptr; jj < stop_ptr; jj++)
               {
                   const int j = stop_ptr - 1 - (jj - start_ptr);
                   const int rowIdx = d_cscRowIdx[j];
                   for (int k = lane_id; k < rhs; k+=WARP_SIZE)
                       atom_add_d_fp64(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
                   mem_fence(CLK_GLOBAL_MEM_FENCE);
                   if (!lane_id)
                       atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
                                             memory_order_acquire, memory_scope_device);
               }
//____________________________BUG_____________________________________
//           else
//           {
//               for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
//               {
//                   const int j = stop_ptr - 1 - (jj - start_ptr);
//                   const int rowIdx = d_cscRowIdx[j];
//                   for (int k = 0; k < rhs; k++)
//                       atom_add_d_fp64(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);
//                   mem_fence(CLK_GLOBAL_MEM_FENCE);
//                   atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,
//                                             memory_order_acquire, memory_scope_device);
//               }
//           }
        }
    }
