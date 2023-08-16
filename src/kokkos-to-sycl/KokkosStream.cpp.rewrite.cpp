// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "KokkosStream.hpp"
#include <sycl/sycl.hpp>

bool cached = false;
std::vector<sycl::device> devices;
void getDeviceList(void);

template <class T>
KokkosStream<T>::KokkosStream(
        const int ARRAY_SIZE, const int device_index)
    : array_size(ARRAY_SIZE),
    a {ARRAY_SIZE},
    b {ARRAY_SIZE},
    c {ARRAY_SIZE},
    sum {1}
{
  if (!cached)
    getDeviceList();

  if (device_index >= devices.size())
    throw std::runtime_error("Invalid device index");

  sycl::device dev = devices[device_index];

  // Print out device information
  std::cout << "Using SYCL device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  // Check device can support FP64 if needed
  if (sizeof(T) == sizeof(double))
  {
    if (!dev.has(sycl::aspect::fp64))
    {
      throw std::runtime_error("Device does not support double precision, please use --float");
    }
  }

  queue = std::make_unique<sycl::queue>(dev, sycl::async_handler{[&](sycl::exception_list l)
  {
    bool error = false;
    for(auto e: l)
    {
      try
      {
        std::rethrow_exception(e);
      }
      catch (sycl::exception e)
      {
        std::cout << e.what();
        error = true;
      }
    }
    if(error)
    {
      throw std::runtime_error("SYCL errors detected");
    }
  }});

  // No longer need list of devices
  devices.clear();
  cached = true;
}

template <class T>
KokkosStream<T>::~KokkosStream()
{

}

template <class T>
void KokkosStream<T>::init_arrays(T initA, T initB, T initC)
{

    queue->submit([&] (sycl::handler &cgh) {
      sycl::accessor accessor_a {a, cgh};





      sycl::accessor accessor_b {b, cgh};





      sycl::accessor accessor_c {c, cgh};

      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_a[index] = initA;
    accessor_b[index] = initB;
    accessor_c[index] = initC;
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
void KokkosStream<T>::read_arrays(
        std::vector<T>& ha, std::vector<T>& hb, std::vector<T>& hc)
{
  sycl::host_accessor _a {a};
  sycl::host_accessor _b {b};
  sycl::host_accessor _c {c};
  for (int i = 0; i < array_size; i++)
  {
    ha[i] = _a[i];
    hb[i] = _b[i];
    hc[i] = _c[i];
  }
}

template <class T>
void KokkosStream<T>::copy()
{

    queue->submit([&] (sycl::handler &cgh) {
      sycl::accessor accessor_a {a, cgh};





      sycl::accessor accessor_b {b, cgh};





      sycl::accessor accessor_c {c, cgh};
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_c[index] = accessor_a[index];
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
void KokkosStream<T>::mul()
{

  const T scalar = startScalar;



    queue->submit([&] (sycl::handler &cgh) {
      sycl::accessor accessor_a {a, cgh};





      sycl::accessor accessor_b {b, cgh};





      sycl::accessor accessor_c {c, cgh};
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_b[index] = scalar*accessor_c[index];
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
void KokkosStream<T>::add()
{










    queue->submit([&] (sycl::handler &cgh) {
      sycl::accessor accessor_a {a, cgh};





      sycl::accessor accessor_b {b, cgh};





      sycl::accessor accessor_c {c, cgh};
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_c[index] = accessor_a[index] + accessor_b[index];
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
void KokkosStream<T>::triad()
{







  const T scalar = startScalar;



    queue->submit([&] (sycl::handler &cgh) {
        sycl::accessor accessor_a {a, cgh};





        sycl::accessor accessor_b {b, cgh};





        sycl::accessor accessor_c {c, cgh};
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_a[index] = accessor_b[index] + scalar*accessor_c[index];
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
void KokkosStream<T>::nstream()
{

  const T scalar = startScalar;



    queue->submit([&] (sycl::handler &cgh) {
      sycl::accessor accessor_a {a, cgh};





      sycl::accessor accessor_b {b, cgh};





      sycl::accessor accessor_c {c, cgh};
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> index) {
    accessor_a[index] += accessor_b[index] + scalar*accessor_c[index];
  });
    });

  ;;



    queue->wait();
  ;;
}

template <class T>
T KokkosStream<T>::dot()
{

  queue->submit([&](sycl::handler &cgh)
  {
    sycl::accessor ka {a, cgh};
    sycl::accessor kb {b, cgh};

    cgh.parallel_for(sycl::range<1>{array_size},
      // Reduction object, to perform summation - initialises the result to zero
      sycl::reduction(sum, cgh, std::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
      [=](sycl::id<1> idx, auto& intermed)
      {
        intermed += ka[idx] * kb[idx];
      });

  });

  // Get access on the host, and return a copy of the data (single number)
  // This will block until the result is available, so no need to wait on the queue.
  sycl::host_accessor result {sum, sycl::read_only};
  return result[0];

}

void getDeviceList(void)
{
  // Ask SYCL runtime for all devices in system
  devices = sycl::device::get_devices();
  cached = true;
}

void listDevices(void)
{
  std::cout << "Kokkos library for " << getDeviceName(0) << std::endl;
}


std::string getDeviceName(const int device)
{
  return "";
}


std::string getDeviceDriver(const int device)
{
  return "Kokkos";
}

template class KokkosStream<float>;
template class KokkosStream<double>;
