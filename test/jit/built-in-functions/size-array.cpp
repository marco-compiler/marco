// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [2, 3]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* x, ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 3 });

	array<long, 2> y = { 0, 0 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	cout << "results" << endl;
	__modelica_ciface_foo(&xDescriptor, &yDescriptor);
	cout << yDescriptor << endl;

	return 0;
}
