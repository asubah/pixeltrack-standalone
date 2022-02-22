#ifndef AlpakaCore_initialise_h
#define AlpakaCore_initialise_h

#include <iostream>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/alpakaDevices.h"
#include "Framework/demangle.h"

namespace cms::alpakatools {

  template <typename TPlatform>
  void initialise() {
    constexpr const char* suffix[] = { "devices.", "device:", "devices:" };

    if (devices<TPlatform>.empty()) {
      devices<TPlatform> = enumerate<TPlatform>();
      auto size = devices<TPlatform>.size();
      //std::cout << edm::demangle<TPlatform> << " platform succesfully initialised." << std::endl;
      std::cout << "Found " << size << " " << suffix[size < 2 ? size : 2] << std::endl;
      for (auto const& device: devices<TPlatform>) {
        std::cout << "  - " << alpaka::getName(device) << std::endl;
      }
    } else {
      //std::cout << edm::demangle<TPlatform> << " platform already initialised." << std::endl;
    }
  }

}  // namespace cms::alpakatools

#endif  // AlpakaCore_initialise_h
