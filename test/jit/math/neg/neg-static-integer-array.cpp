// RUN: marco %p/neg-static-integer-array.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ -g -c %s -I %runtime_h -I %llvm_include_dirs $(llvm-config --cxxflags) -o %basename_t.o
// RUN: clang++ -g %basename_t.o %basename_t.mo.s $(llvm-config --ldflags --libs) -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: -10
// CHECK-NEXT: -23
// CHECK-NEXT: 57

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_negStaticIntegerArray(
		ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 3> x = { 10, 23, -57 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 3> y = { 10, 23, -57 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	__modelica_ciface_negStaticIntegerArray(&xDescriptor, &yDescriptor);

	cout << "results" << endl;

	for (const auto& value : y)
		cout << value << endl;

	return 0;
}
