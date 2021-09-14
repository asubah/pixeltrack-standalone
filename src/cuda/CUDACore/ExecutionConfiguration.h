namespace cms {
  namespace cuda {

    struct ExecutionConfiguration {
      constexpr ExecutionConfiguration() = default;

      constexpr void construct(dim3 grid, dim3 block, size_t shmem, cudaStream_t stream) {
        m_gird = grid;
        m_block = block;
        m_shmem = shmem;
        m_stream = stream;
      }

    private:
      dim3 m_gird;
      dim3 m_block;
      size_t m_shmem;
      cudaStream_t m_stream;
    };

  }  // namespace cuda
}  // namespace cms
