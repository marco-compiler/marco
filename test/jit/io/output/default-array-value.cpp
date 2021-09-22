// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* x);

using namespace std;

int main() {
	array<long, 3> x = { 0, 0, 0 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	__modelica_ciface_foo(&xDescriptor);

	cout << "results" << endl;

	for (const auto& value : xDescriptor)
		cout << value << endl;

	return 0;
}
