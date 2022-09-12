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
    phc->nModules() = nModules;
    phc->nDigis() = nDigis;
  }

  uint32_t nModules() const { return phc->nModules(); }
  uint32_t nDigis() const { return phc->nDigis(); }

  PDC_SiPixelDigis::View view() { return this->pdc.view(); };
  PDC_SiPixelDigis::ConstView const& view() const { return this->pdc.view(); };

//  PHC_SiPixelDigis::View hostView() { return phc.view(); };
//  PHC_SiPixelDigis::ConstView& const hostView() const { return phc.view(); };

  cms::cuda::host::unique_ptr<uint16_t[]> adcToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<int32_t[]> clusToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> pdigiToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> rawIdArrToHostAsync(cudaStream_t stream) const;

private:
  PDC_SiPixelDigis pdc;
  PHC_SiPixelDigis phc;
};

#endif
