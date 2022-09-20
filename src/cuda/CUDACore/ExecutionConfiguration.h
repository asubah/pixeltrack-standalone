#ifndef HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
#define HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h

#include <fstream>

namespace cms {
  namespace cuda {

    class ExecutionConfiguration {
    public:
      const std::string CONFIG_PATH = "src/cuda/CUDACore/kernel_configs/";

      ExecutionConfiguration(){};

      template <typename T>
      void cudaOccCalc(T kernel, int* blockSize, size_t dynamicSMemSize = 0, int blockSizeLimit = 0) {
        int minGridSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, blockSize, kernel, dynamicSMemSize, blockSizeLimit);
      }

      size_t configFromFile(std::string filename) {
        size_t blockSize;

        std::fstream file;
        file.open(CONFIG_PATH + filename, std::ios::in);
        if (file.is_open())
        {
          file >> blockSize;
          file.close();
        }
        // std::cout << "Filename = " << filename << " Blocksize = " << blockSize << "\n";

        return blockSize;
      }
    };

  }  // namespace cuda
}  // namespace cms

#endif  // HeterogeneousCore_CUDAUtilities_interface_ExecutionConfiguration_h
