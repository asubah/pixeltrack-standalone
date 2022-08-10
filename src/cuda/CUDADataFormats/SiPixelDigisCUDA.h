#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "CUDADataFormats/PortableDeviceCollection.h"
#include "CUDADataFormats/PortableHostCollection.h"

#include "DataFormats/SiPixelDigisDeviceLayout.h"
#include "DataFormats/SiPixelDigisHostLayout.h"

class SiPixelDigisCUDA {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t maxFedWords, cudaStream_t stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA &operator=(const SiPixelDigisCUDA &) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA &&) = default;
  SiPixelDigisCUDA &operator=(SiPixelDigisCUDA &&) = default;

  void setNModulesDigis(uint32_t nModules, uint32_t nDigis) {
    hc->nModules() = nModules;
    hc->nDigis() = nDigis;
  }

  uint32_t nModules() const { return hc->nModules(); }
  uint32_t nDigis() const { return hc->nDigis(); }

  PDC_SiPixelDigisDeviceLayout::View deviceView() { return *dc; };
  PDC_SiPixelDigisDeviceLayout::ConstView deviceView() const { return *dc; };
  PDC_SiPixelDigisDeviceLayout::ConstView c_deviceView() const { return *dc; };

  PHC_SiPixelDigisHostLayout::View hostView() { return *hc; };
  PHC_SiPixelDigisHostLayout::ConstView hostView() const { return *hc; };
  PHC_SiPixelDigisHostLayout::ConstView c_hostView() const { return *hc; };

  cms::cuda::host::unique_ptr<uint16_t[]> adcToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<int32_t[]> clusToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> pdigiToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(cudaStream_t stream) const;

private:
  PDC_SiPixelDigisDeviceLayout dc;
  PHC_SiPixelDigisHostLayout hc;
};

#endif
