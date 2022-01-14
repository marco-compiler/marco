// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 9

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(ArrayDescriptor<long, 2>* x);

using namespace std;

int main() {
	array<long, 6> x = { -1, 9, -3, 0, 4, 3 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 3, 2 });

	cout << "results" << endl;
	cout << __modelica_ciface_foo(&xDescriptor)<< endl;

	return 0;
}
