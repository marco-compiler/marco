// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 7.853982e-01
// CHECK-NEXT: 5.235988e-01
// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: -5.235988e-01
// CHECK-NEXT: -7.853982e-01

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" double __modelica_ciface_foo(double x);

using namespace std;

int main() {
	array<double, 5> x = { 1, 0.577350269, 0, -0.577350269, -1 };
	cout << "results" << endl;

	for (const auto& value : x)
		cout << scientific << __modelica_ciface_foo(value) << endl;

	return 0;
}
