
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "OMPStream.h"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
OMPStream<T>::OMPStream(const int ARRAY_SIZE, int device)
{
  array_size = ARRAY_SIZE;

  // Allocate on the host
  this->a = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->b = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);
  this->c = (T*)aligned_alloc(ALIGNMENT, sizeof(T)*array_size);

  omp_set_default_device(device);
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  // Set up data region on device
  #pragma omp target enter data map(alloc: a[0:array_size], b[0:array_size], c[0:array_size])
  {}

}

template <class T>
OMPStream<T>::~OMPStream()
{
  // End data region on device
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target exit data map(release: a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  free(a);
  free(b);
  free(c);
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  int array_size = this->array_size;

  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void OMPStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target update from(a[0:array_size], b[0:array_size], c[0:array_size])
  {}

  #pragma omp parallel for
  for (int i = 0; i < array_size; i++)
  {
    h_a[i] = a[i];
    h_b[i] = b[i];
    h_c[i] = c[i];
  }

}

template <class T>
void OMPStream<T>::copy()
{
  int array_size = this->array_size;
  T *a = this->a;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i];
  }
}

template <class T>
void OMPStream<T>::mul()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    b[i] = scalar * c[i];
  }
}

template <class T>
void OMPStream<T>::add()
{
  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void OMPStream<T>::triad()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void OMPStream<T>::nstream()
{
  const T scalar = startScalar;

  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  T *c = this->c;
  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < array_size; i++)
  {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
T OMPStream<T>::dot()
{
  T sum = 0.0;

  int array_size = this->array_size;
  T *a = this->a;
  T *b = this->b;
  #pragma omp target teams distribute parallel for simd map(tofrom: sum) reduction(+:sum)
  for (int i = 0; i < array_size; i++)
  {
    sum += a[i] * b[i];
  }

  return sum;
}



void listDevices(void)
{
  // Get number of devices
  int count = omp_get_num_devices();

  // Print device list
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << "There are " << count << " devices." << std::endl;
  }
}

std::string getDeviceName(const int)
{
  return std::string("Device name unavailable");
}

std::string getDeviceDriver(const int)
{
  return std::string("Device driver unavailable");
}
template class OMPStream<float>;
template class OMPStream<double>;
