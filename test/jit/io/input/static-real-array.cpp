// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 7.5

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" double __modelica_ciface_foo(ArrayDescriptor<double, 1>* x);

using namespace std;

int main() {
	array<double, 3> x = { 1.5, 2.5, 3.5 };
	ArrayDescriptor<double, 1> xDescriptor(x);

	cout << "results" << endl;
	cout << __modelica_ciface_foo(&xDescriptor) << endl;

	return 0;
}
