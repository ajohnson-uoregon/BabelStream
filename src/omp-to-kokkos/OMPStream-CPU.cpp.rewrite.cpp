
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#include <cstdlib>  // For aligned_alloc
#include "OMPStream.hpp.rewrite.hpp"

#ifndef ALIGNMENT
#define ALIGNMENT (2*1024*1024) // 2MB
#endif

template <class T>
OMPStream<T>::OMPStream(const int ARRAY_SIZE, int device)
{
  Kokkos::initialize();

  array_size = ARRAY_SIZE;

  // Allocate on the host



    device_a = new Kokkos::View<T*>("device_a", array_size);
    hostmirror_a = new typename Kokkos::View<T*>::HostMirror();
    *hostmirror_a = Kokkos::create_mirror_view(*device_a);
  ;;



    device_b = new Kokkos::View<T*>("device_b", array_size);
    hostmirror_b = new typename Kokkos::View<T*>::HostMirror();
    *hostmirror_b = Kokkos::create_mirror_view(*device_b);
  ;;



    device_c = new Kokkos::View<T*>("device_c", array_size);
    hostmirror_c = new typename Kokkos::View<T*>::HostMirror();
    *hostmirror_c = Kokkos::create_mirror_view(*device_c);
  ;;
}

template <class T>
OMPStream<T>::~OMPStream()
{
  Kokkos::finalize();

  ;
  ;
  ;
}

template <class T>
void OMPStream<T>::init_arrays(T initA, T initB, T initC)
{
  int array_size = this->array_size;




    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  });
    Kokkos::fence();


}

template <class T>
void OMPStream<T>::read_arrays(std::vector<T>& h_a, std::vector<T>& h_b, std::vector<T>& h_c)
{



    deep_copy(*hostmirror_a, *device_a);
    deep_copy(*hostmirror_b, *device_b);
    deep_copy(*hostmirror_c, *device_c);

		for (int i = 0; i < array_size; i++) {
    h_a[i] = (*hostmirror_a)[i];
    h_b[i] = (*hostmirror_b)[i];
    h_c[i] = (*hostmirror_c)[i];
  }



}

template <class T>
void OMPStream<T>::copy()
{



    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    c[i] = a[i];
  });
    Kokkos::fence();


}

template <class T>
void OMPStream<T>::mul()
{
  const T scalar = startScalar;




    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    b[i] = scalar * c[i];
  });
    Kokkos::fence();


}

template <class T>
void OMPStream<T>::add()
{



    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    c[i] = a[i] + b[i];
  });
    Kokkos::fence();


}

template <class T>
void OMPStream<T>::triad()
{
  const T scalar = startScalar;




    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    a[i] = b[i] + scalar * c[i];
  });
    Kokkos::fence();


}

template <class T>
void OMPStream<T>::nstream()
{
  const T scalar = startScalar;




    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_for(array_size-0, KOKKOS_LAMBDA (const int i) {
    a[i] += b[i] + scalar * c[i];
  });
    Kokkos::fence();


}

template <class T>
T OMPStream<T>::dot()
{
  T sum = 0.0;




    Kokkos::View<T*> a(*device_a);
    Kokkos::View<T*> b(*device_b);
    Kokkos::View<T*> c(*device_c);

		Kokkos::parallel_reduce(array_size-0, KOKKOS_LAMBDA (const int i, T& intermed) {
    intermed += a[i] * b[i];
  },
		sum);



  return sum;
}



void listDevices(void)
{
  std::cout << "0: CPU" << std::endl;
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
