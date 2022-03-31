// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mod
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [1, 2]
// CHECK-NEXT{LITERAL}: [10, 2]

#include "marco/Runtime/ArrayDescriptor.h"
#include <array>
#include <iostream>

extern "C" long __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* y, ArrayDescriptor<long, 1>* x);

using namespace std;

int main() {
	array<long, 2> x = { 1, 2 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	ArrayDescriptor<long, 1> yDescriptor(nullptr, { 1 });

	cout << "results" << endl;
	__modelica_ciface_foo(&yDescriptor, &xDescriptor);
	cout << xDescriptor << endl;
	cout << yDescriptor << endl;
	free(yDescriptor.getData());

	return 0;
}
