#ifndef HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h

#include <memory>
#include <functional>

#include "CUDACore/allocate_host.h"

namespace cms {
  namespace cuda {
    namespace host {
      namespace impl {

        constexpr size_t default_alignment = 128;

        constexpr size_t pad(size_t size) {
          return ((size + default_alignment - 1) / default_alignment) * default_alignment;
        }

        // Additional layer of types to distinguish device::unique_ptr from host::unique_ptr
        class Deleter {
        public:
          Deleter() = default;  // for edm::Wrapper
          Deleter(bool pinned) : pinned_{pinned} {}

          void operator()(void *ptr) {
            if (pinned_) {
              cms::cuda::free_host(ptr);
            } else {
              std::free(ptr);
            }
          }

        private:
          bool pinned_ = false;
        };
      }  // namespace impl

      template <typename T>
      using unique_ptr = std::unique_ptr<T, impl::Deleter>;

      namespace impl {
        template <typename T>
        struct make_host_unique_selector {
          using non_array = cms::cuda::host::unique_ptr<T>;
        };
        template <typename T>
        struct make_host_unique_selector<T[]> {
          using unbounded_array = cms::cuda::host::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct make_host_unique_selector<T[N]> {
          struct bounded_array {};
        };
      }  // namespace impl
    }    // namespace host

    // Allocate pageable host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique() {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the pageable host memory is not supported");
      void *mem = std::aligned_alloc(host::impl::default_alignment, host::impl::pad(sizeof(T)));
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::Deleter{false}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the pageable host memory is not supported");
      void *mem = std::aligned_alloc(host::impl::default_alignment, host::impl::pad(n * sizeof(element_type)));
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::Deleter{false}};
    }

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique_uninitialized() {
      void *mem = std::aligned_alloc(host::impl::default_alignment, host::impl::pad(sizeof(T)));
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::Deleter{false}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique_uninitialized(size_t n) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = std::aligned_alloc(host::impl::default_alignment, host::impl::pad(n * sizeof(element_type)));
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::Deleter{false}};
    }

    // Allocate pinned host memory
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique(cudaStream_t stream) {
      static_assert(std::is_trivially_constructible<T>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::Deleter{true}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique(size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      static_assert(std::is_trivially_constructible<element_type>::value,
                    "Allocating with non-trivial constructor on the pinned host memory is not supported");
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::Deleter{true}};
    }

    // No check for the trivial constructor, make it clear in the interface
    template <typename T>
    typename host::impl::make_host_unique_selector<T>::non_array make_host_unique_uninitialized(cudaStream_t stream) {
      void *mem = allocate_host(sizeof(T), stream);
      return typename host::impl::make_host_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                          host::impl::Deleter{true}};
    }

    template <typename T>
    typename host::impl::make_host_unique_selector<T>::unbounded_array make_host_unique_uninitialized(
        size_t n, cudaStream_t stream) {
      using element_type = typename std::remove_extent<T>::type;
      void *mem = allocate_host(n * sizeof(element_type), stream);
      return typename host::impl::make_host_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                host::impl::Deleter{true}};
    }

    // Forbid arrays of known bounds (to match std::unique_ptr)
    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique(Args &&...) = delete;

    template <typename T, typename... Args>
    typename host::impl::make_host_unique_selector<T>::bounded_array make_host_unique_uninitialized(Args &&...) = delete;

  }  // namespace cuda
}  // namespace cms

#endif
