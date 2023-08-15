
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "Stream.h"

#include <omp.h>

#define IMPLEMENTATION_STRING "OpenMP"

template <class T>
class OMPStream : public Stream<T>
{
  protected:
    // Size of arrays
    int array_size;

    // Device side pointers
    // T *a;
    // T *b;
    // T *c;

    typename Kokkos::View<T*>* device_a;
    typename Kokkos::View<T*>* device_b;
    typename Kokkos::View<T*>* device_c;
    typename Kokkos::View<T*>::HostMirror* hostmirror_a;
    typename Kokkos::View<T*>::HostMirror* hostmirror_b;
    typename Kokkos::View<T*>::HostMirror* hostmirror_c;

  public:
    OMPStream(const int, int);
    ~OMPStream();

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual void nstream() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;



};