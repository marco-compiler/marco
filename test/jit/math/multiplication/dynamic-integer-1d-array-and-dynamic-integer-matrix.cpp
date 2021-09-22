// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [70, 80, 90]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* z, ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 2>* y);

using namespace std;

int main() {
	array<long, 4> x = { 1, 2, 3, 4 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 12> y = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	ArrayDescriptor<long, 2> yDescriptor(y.data(), { 4, 3 });

	array<long, 3> z = { 0, 0, 0 };
	ArrayDescriptor<long, 1> zDescriptor(z);

	__modelica_ciface_foo(&zDescriptor, &xDescriptor, &yDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
