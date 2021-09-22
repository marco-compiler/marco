// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [1, 2, 3]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 3> y = { 0, 0, 0 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	__modelica_ciface_foo(&yDescriptor);

	cout << "results" << endl;
	cout << yDescriptor << endl;

	return 0;
}
