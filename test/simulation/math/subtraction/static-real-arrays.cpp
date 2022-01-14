// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [-1.000000e+00, 1.000000e+00, 4.000000e+00, -4.000000e+00]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<double, 1>* x, ArrayDescriptor<double, 1>* y, ArrayDescriptor<double, 1>* z);

using namespace std;

int main() {
	array<double, 4> x = { 1.5, -1.5, 1.5, -1.5 };
	ArrayDescriptor<double, 1> xDescriptor(x);

	array<double, 4> y = { 2.5, -2.5, -2.5, 2.5 };
	ArrayDescriptor<double, 1> yDescriptor(y);

	array<double, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<double, 1> zDescriptor(z);

	__modelica_ciface_foo(&xDescriptor, &yDescriptor, &zDescriptor);

	cout << "results" << endl;
	cout << scientific << zDescriptor << endl;

	return 0;
}
