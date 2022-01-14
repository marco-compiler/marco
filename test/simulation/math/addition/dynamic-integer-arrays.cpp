// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [3, -3, -1, 1]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* z, ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 4> x = { 1, -1, 1, -1 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 4> y = { 2, -2, -2, 2 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	ArrayDescriptor<long, 1> zDescriptor(nullptr, { 1 });

	__modelica_ciface_foo(&zDescriptor, &xDescriptor, &yDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;
	free(zDescriptor.getData());

	return 0;
}
