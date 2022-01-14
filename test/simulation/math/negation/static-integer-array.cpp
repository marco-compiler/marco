// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: -10
// CHECK-NEXT: -23
// CHECK-NEXT: 57

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 3> x = { 10, 23, -57 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 3> y = { 10, 23, -57 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	__modelica_ciface_foo(&xDescriptor, &yDescriptor);

	cout << "results" << endl;

	for (const auto& value : yDescriptor)
		cout << value << endl;

	return 0;
}
