// C++ includes
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CUDACore/cudaCheck.h"
#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CondFormats/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(PHC_SiPixelFedCablingMap& cablingMap,
                                                               std::vector<unsigned char> modToUnp)
    : modToUnpDefault(modToUnp.size()), hasQuality_(true) {
  this->cablingMapHost = std::move(cablingMap);
  std::copy(modToUnp.begin(), modToUnp.end(), modToUnpDefault.begin());
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {}

const PDC_SiPixelFedCablingMap::ConstView SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(
    cudaStream_t cudaStream) const {
  const auto& data =
      gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](PDC_SiPixelFedCablingMap& data, cudaStream_t stream) {
        // allocate
        data = PDC_SiPixelFedCablingMap(pixelgpudetails::MAX_SIZE, stream);

        // transfer
        cudaCheck(cudaMemcpyAsync(data.buffer().get(),
                                  this->cablingMapHost.buffer().get(),
                                  this->cablingMapHost.bufferSize(),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.view();
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault;
}

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(cudaFree(modToUnpDefault)); }
