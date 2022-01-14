// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [14, 32, 50, 68]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* x, ArrayDescriptor<long, 1>* y, ArrayDescriptor<long, 1>* z);

using namespace std;

int main() {
	array<long, 12> x = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 4, 3 });

	array<long, 3> y = { 1, 2, 3 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	array<long, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<long, 1> zDescriptor(z);

	__modelica_ciface_foo(&xDescriptor, &yDescriptor, &zDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
