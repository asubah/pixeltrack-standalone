#include "CUDADataFormats/SiPixelDigisCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream) {
  this->pdc = PDC_SiPixelDigis(maxFedWords, stream);
  this->phc = PHC_SiPixelDigis(maxFedWords, stream);
}

cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), pdc->adc(), nDigis() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), pdc->clus(), nDigis() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), pdc->pdigi(), nDigis() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), pdc->rawIdArr(), nDigis() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  return ret;
}
