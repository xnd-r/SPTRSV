#include "include/common.h"
#include "include/utils.h"
#include "include/basiccl.h"

double sptrsv_syncfree3_opencl (int           *csrColIdx,
                               int           *csrRowPtr,
                               const VALUE_TYPE    *csrVal,
                               const int            m,
                               const int            n,
                               const int            nnz,
                               VALUE_TYPE    *x,
                               const VALUE_TYPE    *b)
{
    const int rhs = 1;
    const int device_id = 1;
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    int err = 0;

    // set device
    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices = NULL;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    ocl_command_queue;      // OpenCL Gpu command queues

    bool profiling = true;

    // platform
    err = basicCL.getNumPlatform(&numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL getNumPlatform ERROR CODE = %i\n", err); return err;}
    DEBUG_INFO("Platform number: %i.\n", numPlatforms);

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL getPlatformIDs ERROR CODE = %i\n", err); return err;}

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) {printf("OpenCL getPlatformInfo ERROR CODE = %i\n", err); return err;}

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[device_id], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL getDeviceInfo ERROR CODE = %i\n", err); return err;}

            DEBUG_INFO("Platform [%i] Vendor: %s Version: %s\n", i, platformVendor, platformVersion);
            DEBUG_INFO("Using GPU device: %s ( %i CUs, %lu kB local, %lu MB global, %s \n)",
                   gpuDeviceName, gpuDeviceComputeUnits,
                   gpuDeviceLocalMem / 1024, gpuDeviceGlobalMem / (1024 * 1024), gpuDeviceVersion);

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Gpu commandqueue
    if (profiling)
        err = basicCL.getCommandQueueProfilingEnable(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    else
        err = basicCL.getCommandQueue(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    FILE* fp;
    char* source_str;
    size_t source_size;
    fp = fopen("./src/kernel_full3.cl", "r");
    if (!fp) {
        DEBUG_INFO("Failed to load kernel. Exit\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_str[MAX_SOURCE_SIZE + 1] = '\0';
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);


    // Create the program
    cl_program          ocl_program_sptrsv;


    ocl_program_sptrsv = clCreateProgramWithSource(cxGpuContext, 1, (const char**)&source_str, (const size_t*)&source_size, &err);

    if(err != CL_SUCCESS) {printf("OpenCL clCreateProgramWithSource ERROR CODE = %i\n", err); return err;}

    // Build the program

    err = clBuildProgram(ocl_program_sptrsv, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=double", NULL, NULL);
    if (err != CL_SUCCESS) {
      printf("Error: clBuildProgram() returned %d.\n", err);
      size_t buildLogSize = 0;
      clGetProgramBuildInfo(ocl_program_sptrsv, cdGpuDevices[device_id],
                            CL_PROGRAM_BUILD_LOG, 0, NULL, &buildLogSize);
      // TODO: Fix BuildLog
      printf("%zu\n", buildLogSize);
      cl_char *buildLog = new cl_char[buildLogSize];
      if (buildLog) {
        clGetProgramBuildInfo(ocl_program_sptrsv, cdGpuDevices[device_id],
                              CL_PROGRAM_BUILD_LOG, buildLogSize, buildLog,
                              NULL);
        printf(">>> Build Log:\n");
        printf("%s\n", buildLog);
        printf("<<< End of Build Log\n");
        std::cout << buildLog << std::endl;
      }
      exit(0);
    }

    cl_kernel  ocl_kernel_sptrsv_executor;
    ocl_kernel_sptrsv_executor = clCreateKernel(ocl_program_sptrsv, "sptrsv_syncfree_opencl_executor", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel1 ERROR CODE = %i\n", err); return err;}

    // transfer host mem to device mem
    // Define pointers of matrix L, vector x and b
    cl_mem      d_csrRowPtr;
    cl_mem      d_csrColIdx;
    cl_mem      d_csrVal;
    cl_mem      d_b;
    cl_mem      d_x;

    // Matrix L
    d_csrRowPtr = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (m+1) * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_csrColIdx = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnz  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_csrVal    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnz  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrRowPtr, CL_TRUE, 0, (m+1) * sizeof(int), csrRowPtr, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrColIdx, CL_TRUE, 0, nnz  * sizeof(int), csrColIdx, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_csrVal, CL_TRUE, 0, nnz  * sizeof(VALUE_TYPE), csrVal, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector b
    d_b    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, m * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_b, CL_TRUE, 0, m * sizeof(VALUE_TYPE), b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector x
    d_x    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    memset(x, 0, m  * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    cl_mem d_get_value;
    d_get_value = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    int *get_value = (int *)malloc(m * sizeof(int));
    memset(get_value, 0, m * sizeof(int));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_get_value, CL_TRUE, 0, m  * sizeof(int), get_value, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}



    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads;
    int num_blocks;

    // step 5: solve L*y = x
    //const int wpb = WARP_PER_BLOCK;

    err  = clSetKernelArg(ocl_kernel_sptrsv_executor, 0,  sizeof(cl_mem), (void*)&d_csrRowPtr);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 1,  sizeof(cl_mem), (void*)&d_csrColIdx);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 2,  sizeof(cl_mem), (void*)&d_csrVal);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 3,  sizeof(cl_mem), (void*)&d_get_value);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 4,  sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 5,  sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 6,  sizeof(cl_mem), (void*)&d_x);


    double time_opencl_solve = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {

        // memset d_get_value to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_get_value, CL_TRUE, 0, m * sizeof(int), get_value, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        err = clEnqueueWriteBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        num_threads = WARP_PER_BLOCK * WARP_SIZE;
        num_blocks = ceil ((double)m / (double)(num_threads));
        szLocalWorkSize[0]  = num_threads;
        szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsv_executor, 1,
                                         NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsv_executor kernel run error = %i\n", err); return err; }


        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        time_opencl_solve += double(endTime - startTime) / 1000000.0;
    }

    time_opencl_solve /= BENCH_REPEAT;

    err = clEnqueueReadBuffer(ocl_command_queue, d_x, CL_TRUE, 0, m * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}


    // step 6: free resources
    free(get_value);

    if(d_get_value) err = clReleaseMemObject(d_get_value); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}


    if(d_csrRowPtr) err = clReleaseMemObject(d_csrRowPtr); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrColIdx) err = clReleaseMemObject(d_csrColIdx); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_csrVal)    err = clReleaseMemObject(d_csrVal); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_b) err = clReleaseMemObject(d_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_x) err = clReleaseMemObject(d_x); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    return time_opencl_solve / 1000.;
}
