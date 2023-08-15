#include "ClangRewriteMacros.h"

#include <Kokkos_Core.hpp>
#include <sycl/sycl.hpp>

std::vector<std::string> clang_rewrite_literal_names = {"parallel_for", "parallel_reduce", "queue", "fence", "T", "cgh"};

template <class T>
class FakeClass {
  Kokkos::View<T*> *d_a;

  auto view_m();
};

template <class T>
[[clang::matcher("view")]]
auto FakeClass<T>::view_m() {
  [[clang::matcher_block]] {
    Kokkos::View<T*> a(*d_a);
  }
}

template <class T>
[[clang::replace("view")]]
auto view_r() {
  [[clang::rewrite_setup]]
    sycl::buffer<T> a;
  [[clang::rewrite_setup]]
    sycl::accessor ka = clang_rewrite::code_literal("accessor_" + clang_rewrite::to_str(a));

  [[clang::matcher_block]] {
    sycl::accessor ka {a};
  }
}

[[clang::matcher("fence")]]
auto fence_m() {
  [[clang::matcher_block]] {
    Kokkos::fence();
  }
}

[[clang::replace("fence")]]
auto fence_r() {
  [[clang::rewrite_setup]]
    std::unique_ptr<sycl::queue> queue;

  [[clang::matcher_block]] {
    queue->wait();
  }
}




template <class T>
[[clang::matcher("kokkos")]]
auto kokkos_m() {
  [[clang::rewrite_setup]]
    int array_size;

  [[clang::matcher_block]] {
    Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (const T idx) {
      clang_rewrite::loop_body();
    });
  }
}

template <class T>
[[clang::replace("kokkos")]]
auto kokkos_r() {
  [[clang::rewrite_setup]]
    size_t array_size;
  [[clang::rewrite_setup]]
    std::unique_ptr<sycl::queue> queue;

  [[clang::matcher_block]] {
    queue->submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{array_size}, [=](sycl::id<1> idx) {
        clang_rewrite::loop_body();
      });
    });

  }
}

template <class T>
[[clang::matcher("reduction")]]
auto red_m() {
  [[clang::rewrite_setup]]
    int array_size, red;

  [[clang::matcher_block]] {
    Kokkos::parallel_reduce(array_size, KOKKOS_LAMBDA (const int idx, T& intermed) {
			clang_rewrite::loop_body();
		},
		red);
  }
}

template <class T>
[[clang::replace("reduction")]]
auto red_r() {
  [[clang::rewrite_setup]]
    int array_size, red;

  [[clang::matcher_block]] {
    queue->submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::range<1>{array_size},
        sycl::reduction(red, cgh, std::plus<T>(), sycl::property::reduction::initialize_to_identity{}),
        [=](sycl::id<1> idx, auto& intermed) {
          clang_rewrite::loop_body();
        });
    });

  }
}
