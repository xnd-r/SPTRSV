#pragma once
#include "CL/opencl.h"

#define CL_STRING_LENGTH 128

class BasicCL
{
public:
    BasicCL();

    int getNumPlatform(cl_uint *numPlatforms);
    int getPlatformIDs(cl_platform_id *platforms, cl_uint numPlatforms);
    int getPlatformInfo(cl_platform_id platform, char *platformVendor, char *platformVersion);

    int getNumCpuDevices(cl_platform_id platform, cl_uint *numCpuDevices);
    int getNumGpuDevices(cl_platform_id platform, cl_uint *numGpuDevices);
    int getCpuDeviceIDs(cl_platform_id platform, cl_uint numCpuDevices, cl_device_id *cpuDevices);
    int getGpuDeviceIDs(cl_platform_id platform, cl_uint numGpuDevices, cl_device_id *gpuDevices);
    int getDeviceInfo(cl_device_id device, char *deviceName, char *deviceVersion,
                      int *deviceComputeUnits, cl_ulong *deviceGlobalMem,
                      cl_ulong *deviceLocalMem, int *maxSubDevices);

    int getContext(cl_context *context, cl_device_id *devices, cl_uint numDevices);

    int getCommandQueue(cl_command_queue *commandQueue, cl_context context, cl_device_id device);
    int getCommandQueueProfilingEnable(cl_command_queue *commandQueue, cl_context context, cl_device_id device);

    int getProgram(cl_program *program, cl_context context, const char *kernelSourceCode);
    int getProgramFromFile(cl_program *program, cl_context context, const char *sourceFilename);
    int getKernel(cl_kernel *kernel, cl_program program, const char *kernelName);
    int getEventTimer(cl_event event,
                      cl_ulong *queuedTime, cl_ulong *submitTime,
                      cl_ulong *startTime,  cl_ulong *endTime);

private:
    cl_int  _ciErr;
};
