// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 6

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" double __modelica_ciface_foo(ArrayDescriptor<double, 1>* x, long i);

using namespace std;

int main() {
	array<double, 3> x = { 1, 2, 3 };
	ArrayDescriptor<double, 1> xDescriptor(x);

	long i = 1;

	cout << "results" << endl;
	cout << __modelica_ciface_foo(&xDescriptor, i) << endl;

	return 0;
}
