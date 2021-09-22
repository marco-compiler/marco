// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 1
// CHECK-NEXT: 3
// CHECK-NEXT: 21
// CHECK-NEXT: -19

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(long x);

using namespace std;

int main() {
	array<long, 4> x = { 0, 1, 10, -10 };

	cout << "results" << endl;

	for (const auto& value : x)
		cout << __modelica_ciface_foo(value) << endl;

	return 0;
}
