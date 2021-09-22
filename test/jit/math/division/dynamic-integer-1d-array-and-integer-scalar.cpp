// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [5, -1, 0]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* z, ArrayDescriptor<long, 1>* x, long y);

using namespace std;

int main() {
	array<long, 3> x = { 10, -2, 0 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	long y = 2;

	array<long, 3> z = { 0, 0, 0 };
	ArrayDescriptor<long, 1> zDescriptor(z);

	__modelica_ciface_foo(&zDescriptor, &xDescriptor, y);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
