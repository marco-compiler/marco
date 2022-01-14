// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* z, ArrayDescriptor<long, 2>* x, ArrayDescriptor<long, 2>* y);

using namespace std;

int main() {
	array<long, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 3 });

	array<long, 6> y = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<long, 2> yDescriptor(y.data(), { 3, 2 });

	array<long, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<long, 2> zDescriptor(z.data(), { 2, 2 });

	__modelica_ciface_foo(&zDescriptor, &xDescriptor, &yDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
