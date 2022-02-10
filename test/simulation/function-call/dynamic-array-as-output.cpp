// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mod
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [2.000000e+00, 4.000000e+00, 6.000000e+00]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<double, 1>* y);

using namespace std;

int main() {
	array<double, 3> y = { 0, 0, 0 };
	ArrayDescriptor<double, 1> yDescriptor(y);

	__modelica_ciface_foo(&yDescriptor);

	cout << "results" << endl;
	cout << scientific << yDescriptor << endl;

	return 0;
}
