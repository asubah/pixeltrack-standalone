#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <cuda_runtime.h>

#include "CUDADataFormats/PortableDeviceCollection.h"
#include "CUDADataFormats/PortableHostCollection.h"

#include "DataFormats/PixelErrorsSoA.h"

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cudaStream_t stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  auto error() { return *error_d; }
  auto const error() const { return *error_d; }
  auto const c_error() const { return *error_d; }

  void copyErrorToHostAsync(cudaStream_t stream);

private:
  PortableDeviceCollection<PixelErrorCompactSoALayout<>> error_d;
  PortableHostCollection<PixelErrorCompactSoALayout<>> error_h;

  PixelFormatterErrors formatterErrors_h;
};

#endif
