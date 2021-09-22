// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* y, long x);

using namespace std;

int main() {
	long x = 2;
	ArrayDescriptor<long, 1> yDescriptor(nullptr, { 1 });

	__modelica_ciface_foo(&yDescriptor, x);

	cout << "results" << endl;

	for (const auto& value : yDescriptor)
		cout << value << endl;

	return 0;
}
