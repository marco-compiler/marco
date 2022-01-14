// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: -1.000000e+01
// CHECK-NEXT: -2.300000e+01
// CHECK-NEXT: 5.700000e+01

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<double, 1>* y, ArrayDescriptor<double, 1>* x);

using namespace std;

int main() {
	array<double, 3> x = { 10, 23, -57 };
	ArrayDescriptor<double, 1> xDescriptor(x);

	array<double, 3> y = { 10, 23, -57 };
	ArrayDescriptor<double, 1> yDescriptor(y);

	__modelica_ciface_foo(&yDescriptor, &xDescriptor);

	cout << "results" << endl;

	for (const auto& value : yDescriptor)
		cout << scientific << value << endl;

	free(yDescriptor.getData());
	return 0;
}
