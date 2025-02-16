#ifndef HeterogeneousCore_SYCLCore_chooseDevice_h
#define HeterogeneousCore_SYCLCore_chooseDevice_h

#include <CL/sycl.hpp>

#include "Framework/Event.h"

namespace cms::sycltools {
  std::vector<sycl::device> const& enumerateDevices(bool verbose = false);
  std::vector<sycl::platform> const& enumeratePlatforms(bool verbose = false);
  sycl::device chooseDevice(edm::StreamID id, bool verbose = false);
}  // namespace cms::sycltools

#endif
