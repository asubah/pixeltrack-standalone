#ifndef DataFormats_interface_PortableHostCollection_h
#define DataFormats_interface_PortableHostCollection_h

#include <cstdlib>

#include "CUDACore/host_unique_ptr.h"

// generic SoA-based product in host memory
template <typename T>
class PortableHostCollection {
public:
  using Layout = T;
  using View = typename Layout::View;
  using Buffer = cms::cuda::host::unique_ptr<std::byte[]>;

  PortableHostCollection() = default;

  PortableHostCollection(int32_t elements)
      // allocate pageable host memory
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements))},
        layout_{buffer_.get(), elements},
        view_{layout_} {
    // make_host_unique for pageable host memory uses a default alignment of 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
  }

  PortableHostCollection(int32_t elements, cudaStream_t stream)
      // allocate pinned host memory, accessible by the current device
      : buffer_{cms::cuda::make_host_unique<std::byte[]>(Layout::computeDataSize(elements), stream)},
        layout_{buffer_.get(), elements},
        view_{layout_} {
    // CUDA pinned host memory uses a default alignment of at least 128 bytes
    assert(reinterpret_cast<uintptr_t>(buffer_.get()) % Layout::alignment == 0);
  }

  // non-copyable
  PortableHostCollection(PortableHostCollection const &) = delete;
  PortableHostCollection &operator=(PortableHostCollection const &) = delete;

  // movable
  PortableHostCollection(PortableHostCollection &&other) = default;
  PortableHostCollection &operator=(PortableHostCollection &&other) = default;

  // default destructor
  ~PortableHostCollection() = default;

  // access the View
  View &view() { return view_; } 
  View const &view() const { return view_; } 

  View &operator*() { return view_; }
  View const &operator*() const { return view_; }

  View *operator->() { return &view_; }
  View const *operator->() const { return &view_; }

  Buffer &buffer() { return buffer_; }
  Buffer const &buffer() const { return buffer_; }

private:
  Buffer buffer_;
  Layout layout_;
  View view_;
};

#endif  // DataFormats_interface_PortableHostCollection_h
