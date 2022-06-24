#include "CUDADataFormats/SiPixelDigiErrorsCUDA.h"

#include "CUDACore/copyAsync.h"
#include "CUDACore/memsetAsync.h"

#include <cassert>

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream)
    : formatterErrors_h(std::move(errors)) {

  this->error_d = PortableDeviceCollection<PixelErrorCompactSoALayout<>>(maxFedWords, stream);
  this->error_h = PortableHostCollection<PixelErrorCompactSoALayout<>>(maxFedWords, stream);

  cms::cuda::memsetAsync(error_d.buffer(), 0x00, error_d.bufferSize(), stream);
}

void SiPixelDigiErrorsCUDA::copyErrorToHostAsync(cudaStream_t stream) {
  cms::cuda::copyAsync(error_h.buffer(), error_d.buffer(), error_d.bufferSize(), stream);
}
