#include "CUDADataFormats/SiPixelClustersCUDA.h"

#include "CUDACore/device_unique_ptr.h"
#include "CUDACore/host_unique_ptr.h"
#include "CUDACore/copyAsync.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters, cudaStream_t stream) {
    this->pdc = PDC_SiPixelClustersCUDA(maxClusters, stream);
}
