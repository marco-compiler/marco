// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* y, ArrayDescriptor<long, 1>* x);

using namespace std;

int main() {
	array<long, 3> x = { 1, 2, 3 };
	ArrayDescriptor<long, 1> xDescriptor(x);
	ArrayDescriptor<long, 2> yDescriptor(nullptr, { 1, 1 });

	__modelica_ciface_foo(&yDescriptor, &xDescriptor);

	cout << "results" << endl;
	cout << yDescriptor << endl;
	free(yDescriptor.getData());

	return 0;
}