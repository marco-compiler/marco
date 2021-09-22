// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: false
// CHECK-NEXT: true

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<bool, 1>* x, ArrayDescriptor<bool, 1>* y);

using namespace std;

int main() {
	array<bool, 2> x = { true, false };
	ArrayDescriptor<bool, 1> xDescriptor(x);

	array<bool, 2> y = { true, false };
	ArrayDescriptor<bool, 1> yDescriptor(y);

	__modelica_ciface_foo(&xDescriptor, &yDescriptor);

	cout << "results" << endl;

	for (const auto& value : yDescriptor)
		cout << boolalpha << value << endl;

	return 0;
}
