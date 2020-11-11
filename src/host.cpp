/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/********************************************************************************************
 * Description:
 * This is host application to test HBM Full bandwidth.
 * Design contains 8 compute units of Kernel. Each compute unit has full access
 *to all HBM
 * memory (0 to 31). Host application allocate buffers into all 32 HBM Banks(16
 *Input buffers
 * and 16 output buffers). Host application runs all 8 compute units together
 *and measure
 * the overall HBM bandwidth.
 *
 ******************************************************************************************/

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"
#include "uniform-int-distribution.hpp"

#define NUM_KERNEL 1

// HBM Banks requirements
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};


double random_chasing_test(unsigned int dataSize, cl::Context context, cl::Kernel krnl, cl::CommandQueue q){
  //generate the random test array
  std::vector<unsigned int, aligned_allocator<unsigned int>> randomArray(dataSize);
  for(unsigned int i = 0; i < dataSize; i++){
    randomArray[i] = i;
  }

  //shuffle the array
  UniformIntDistribution uniform;
  for(unsigned int i = 0; i < dataSize; i++){
    unsigned int j = i + uniform.draw(dataSize - i);
    if(i != j) std::swap(randomArray[i], randomArray[j]);
  }
  
  
  unsigned int start_addr = 0;
  unsigned int addr_sw = start_addr;
  
  for(unsigned int i = 0;i < dataSize; i++){
    addr_sw = randomArray[addr_sw];
  }
  
  cl_int err;
  cl_mem_ext_ptr_t inBufExt1;
  cl_mem_ext_ptr_t inBufExt2;

  cl::Buffer arrayBuffer;
  cl::Buffer addrBuffer;

  
  inBufExt1.obj = randomArray.data();
  inBufExt1.param = 0;
  inBufExt1.flags = bank[0];

  inBufExt2.obj = &start_addr;
  inBufExt2.param = 0;
  inBufExt2.flags = bank[0];
  
  OCL_CHECK(err, arrayBuffer = cl::Buffer(
                       context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX |
                                    CL_MEM_USE_HOST_PTR,
                       sizeof(uint32_t) * dataSize, &inBufExt1, &err));

  OCL_CHECK(err, addrBuffer = cl::Buffer(
                       context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX |
                                    CL_MEM_USE_HOST_PTR,
                       sizeof(uint32_t) * 1, &inBufExt2, &err)); 
 
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                       {arrayBuffer,addrBuffer},
                       0 /* 0 means from host*/));
  q.finish();

  double kernel_time_in_sec = 0;

  std::chrono::duration<double> kernel_time(0);

  auto kernel_start = std::chrono::high_resolution_clock::now();
  OCL_CHECK(err, err = krnl.setArg(0, arrayBuffer));
  OCL_CHECK(err, err = krnl.setArg(1, dataSize));
  OCL_CHECK(err, err = krnl.setArg(2, addrBuffer));

    // Invoking the kernel
  OCL_CHECK(err, err = q.enqueueTask(krnl));

  q.finish();

  auto kernel_end = std::chrono::high_resolution_clock::now();

  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

  kernel_time_in_sec = kernel_time.count();
  
  double kernel_time_in_ms = kernel_time_in_sec * 1000; 
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
                      {addrBuffer},
                       CL_MIGRATE_MEM_OBJECT_HOST));
  q.finish();

  std::cout << "Total memory access is " << dataSize << " times" <<std::endl;
  std::cout << "Total time = "<< kernel_time_in_ms << "ms" <<std::endl;
  std::cout << "The final addr after pointer-chasing is " << start_addr << std::endl;
  std::cout << "The final addr_sw is " << addr_sw << std::endl;
  
  return kernel_time_in_ms;
}


int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <XCLBIN> \n", argv[0]);
    return -1;
  }

  std::string binaryFile = argv[1];
  cl_int err;
  cl::CommandQueue q;
  std::string krnl_name = "pointer_chasing";
  cl::Kernel krnls;
  cl::Context context;

  // OPENCL HOST CODE AREA START
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();

  // read_binary_file() command will find the OpenCL binary file created using
  // the
  // V++ compiler load into OpenCL Binary and return pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);

  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                                            CL_QUEUE_PROFILING_ENABLE,
                                        &err));

    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      // Creating Kernel object using Compute unit names

        // Here Kernel object is created by specifying kernel name along with
        // compute unit.
        // For such case, this kernel object can only access the specific
        // Compute unit

      OCL_CHECK(err, krnls = cl::Kernel(program, krnl_name.c_str(), &err));
    }
  }

  random_chasing_test(1024, context, krnls, q);
  random_chasing_test(2048, context, krnls, q);
  random_chasing_test(4096, context, krnls, q);
  random_chasing_test(8192, context, krnls, q);
  return (EXIT_SUCCESS);

}
