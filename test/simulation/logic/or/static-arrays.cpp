// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<bool, 1>* x, ArrayDescriptor<bool, 1>* y, ArrayDescriptor<bool, 1>* z);

using namespace std;

int main() {
	array<bool, 4> x = { false, false, true, true };
	ArrayDescriptor<bool, 1> xDescriptor(x);

	array<bool, 4> y = { false, true, false, true };
	ArrayDescriptor<bool, 1> yDescriptor(y);

	array<bool, 4> z = { true, true, true, false };
	ArrayDescriptor<bool, 1> zDescriptor(z);

	__modelica_ciface_foo(&xDescriptor, &yDescriptor, &zDescriptor);

	cout << "results" << endl;

	for (const auto& value : zDescriptor)
		cout << boolalpha << value << endl;

	return 0;
}