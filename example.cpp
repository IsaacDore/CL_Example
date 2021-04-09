//https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/

#pragma comment(lib, "OpenCL.lib") //search for this library 
#include "cl.h"
#include "iostream"

const char* k =									//This is a program. A program can contain multiple kernels. A kernel is essentially a function
"__kernel void AddArrays(__global float *a,"	//do not forget double underscores
"__global float *b,"							
"__global float *result,"
"const unsigned int ArraySize)"
"{"
"int id = get_global_id(0);"					//get_global_id(0): gets the id of this work item. 0 means 1D, 1 means 2D, 2 means 3D, can't do over 3D 
"if(id<ArraySize){result[id] = a[id] + b[id];}"	
"}";

int main() {
	unsigned int Sz = 50;				//size of the arrays
	float* a = new float[Sz];			//the two arrays we will add together
	float* b = new float[Sz];
	for (int i = 0; i < Sz; i++) {
		a[i] = i;						//setting their values
		b[i] = i * i;
	}
	float* result = new float[Sz];		//the array in which the result will be copied

	cl_mem _a, _b, _result;				//memory objects: numbers reserved to identify an array in device memory

	cl_platform_id platform_id;			//the platform is the openCl implementation (that of Nvidia vs that of AMD)
	cl_context context;					//a context is a set of devices
	cl_device_id device_id;				//the device is the actual cpu/gpu/...
	cl_command_queue queue;				//a list of tasks to be completed
	cl_program program;					//a compiled program (in this case k above)
	cl_kernel kernel;					//the specific kernel that we will execute within our program

	cl_int error;											//where we store error codes https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
	size_t globalSize, localSize = 64;						//globalSize is the amount of number that we add together. localSize is the size of a work group. globalSize needs to be multiple of localSize
															//globalSize vs localSize is number of soldier in army vs in single battalion
	globalSize = ceilf((float)Sz / localSize)*localSize;	//ensures that globalSize is multiple of localSize

	error = clGetPlatformIDs(1, &platform_id, NULL);								//returns an error code. (lenght of platform array, array of platforms, pointer to unsigned int where function stores number of platforms found) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetPlatformIDs.html
	//std::cout << error << '\n';
	error = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);	//returns an error code. (the specified platform, the device type |Cpu/Gpu/...|, lenght of device array, array of devices, pointer to unsigned int where function stores number of devices found) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clGetDeviceIDs.html
	//std::cout << error << '\n';
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &error);				//returns a newly created context. (properties, lenght of device array, array of devices used in this context, callback function for errors, "userdata" argument of callback function, a pointer to the variable in wich the function stores the error code) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateContext.html 				
	//std::cout << error << '\n';
	queue = clCreateCommandQueueWithProperties(context, device_id, 0, &error);		//returns a newly created command queue. (the context of this command queue, the device that will perform the command, properties, a pointer to the variable in wich the function stores the error code) https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/clCreateCommandQueueWithProperties.html
	//std::cout << error << '\n';
	program = clCreateProgramWithSource(context, 1, &k, NULL, &error);				//returns a newly created program object. (the context of the program, the number of string that make up the source code, an array of those strings |char**|, the lenght of those strings |optional|, a pointer to the variable in wich the function stores the error code) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateProgramWithSource.html
	//std::cout << error << '\n';
	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);						//return an error code. (the program object that will be built, lenght of device array, array of devices for which the program is built |if null then built for all devices in context|, options, callback function that is called when program is built, "userdata" argument of callback function) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clBuildProgram.html
	//std::cout << error << '\n';
	kernel = clCreateKernel(program, "AddArrays", &error);							//returns a newly created kernel object. (the program from which the kernel originates, the name of the kernel |in the source code|, a pointer to the variable in wich the function stores the error code) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateKernel.html
	//std::cout << error << '\n';

	_a = clCreateBuffer(context, CL_MEM_READ_ONLY, Sz * sizeof(float), NULL, NULL);			//reserves a chunk of memory to be used as a buffer. returns a memory object. (the context of this buffer, flags that describe how buffer is used, size in bytes of buffer, |host_ptr||?| just make it NULL, a pointer to the variable in wich the function stores the error code) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateBuffer.html
	_b = clCreateBuffer(context, CL_MEM_READ_ONLY, Sz * sizeof(float), NULL, NULL);
	_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Sz * sizeof(float), NULL, NULL);

	error = clEnqueueWriteBuffer(queue, _a, CL_TRUE, 0, Sz * sizeof(float), a, 0, NULL, NULL);		//writes to a given buffer in memory. returns an error code. (the queue to which the write command will be sent, the buffer in which we write, idk, offset |where we start writing in the buffer|, size in bytes of data, the array that will be copied to the buffer, just make it NULL, idem, idem) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueWriteBuffer.html
	error |= clEnqueueWriteBuffer(queue, _b, CL_TRUE, 0, Sz * sizeof(float), b, 0, NULL, NULL);		// |= because error code probably redundant
	//std::cout << error << '\n';

	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &_a);			//sets a kernel argument. returns an error code.(the kernel, the argument index |0 for first, 1 for second, etc|, size of the argument value in bytes (if memory object, size of memory object), argument value) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clSetKernelArg.html 
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &_b);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &_result);
	error |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &Sz);
	//std::cout << error << '\n';

	error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);	//adds kernel to command queue. returns an error code. (the queue in which the command is added, the kernel that is added, the number of dimensions, make this NULL, total number of work items per dimensions, number of work items per work group per dimensions, just make this null, idem) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueNDRangeKernel.html
	//std::cout << error << '\n';

	clFinish(queue); //wait for queue to finish. returns error code. (queue) https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clFinish.html

	clEnqueueReadBuffer(queue, _result, CL_TRUE, 0, sizeof(float)*Sz, result, 0, NULL, NULL); //reads from a given buffer in memory, returns an error code. (the queue to which the read command will be sent, the buffer from which we read, idk, offset |where we start reading in the buffer|, size in bytes of data, the array to which the buffer is copied, just make it NULL, idem, idem)

	for (int i = 0; i < Sz; i++) {
		std::cout << result[i] << '\n';		//displaying results
	}
	
	clReleaseMemObject(_a);			//Delete what is now unused
	clReleaseMemObject(_b);
	clReleaseMemObject(_result);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	delete[] a;
	delete[] b;
	delete[] result;

	system("pause");
	return 0;
}