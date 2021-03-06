#ifndef GLOBAL_COMMON_HPP_
#define GLOBAL_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/video.hpp>

#include <cmath>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <ctype.h>
#include <sstream>
#include <algorithm>

using namespace std;

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFAGS_H_ to detect if it is version
// 2.1. If yes , we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

#ifndef CPU_ONLY

#if 0
 CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << itf::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << itf::curandGetErrorString(status); \
  } while (0)

 CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

 CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
#endif

#endif  // CPU_ONLY

namespace itf {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// Common functions and classes from std that itf often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
//void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Itf {
 public:
  ~Itf();

  /*
  inline static Itf& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Itf());
    }
    return *singleton_;
  }
  */
  //enum Brew { CPU, GPU };
  //enum Phase { TRAIN, TEST };


#ifndef CPU_ONLY
  //inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  //inline static curandGenerator_t curand_generator() {
  //  return Get().curand_generator_;
  //}
#endif

  // Returns the mode: running on CPU or GPU.
  //inline static Brew mode() { return Get().mode_; }
  // Returns the phase: TRAIN or TEST.
  //inline static Phase phase() { return Get().phase_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  //inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the phase.
  //inline static void set_phase(Phase phase) { Get().phase_ = phase; }
  // Sets the random seed of both boost and curand
  //static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  //static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void ITF_Logging();

 protected:
#ifndef CPU_ONLY
  //cublasHandle_t cublas_handle_;
  //curandGenerator_t curand_generator_;
#endif
  //shared_ptr<RNG> random_generator_;

  //Brew mode_;
  //Phase phase_;
  //static shared_ptr<Itf> singleton_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Itf();

  DISABLE_COPY_AND_ASSIGN(Itf);
};

#ifndef CPU_ONLY

// NVIDIA_CUDA-5.5_Samples/common/inc/helper_cuda.h
//const char* cublasGetErrorString(cublasStatus_t error);
//const char* curandGetErrorString(curandStatus_t error);

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if 0
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
#endif


#endif  // CPU_ONLY

}  // namespace itf

#endif  // GLOBAL_COMMON_HPP_
