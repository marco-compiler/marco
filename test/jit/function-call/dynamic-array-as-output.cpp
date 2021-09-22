// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [2.000000e+00, 4.000000e+00, 6.000000e+00]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<double, 1>* y);

using namespace std;

int main() {
	array<double, 3> y = { 0, 0, 0 };
	ArrayDescriptor<double, 1> yDescriptor(y);

	__modelica_ciface_foo(&yDescriptor);

	cout << "results" << endl;
	cout << scientific << yDescriptor << endl;

	return 0;
}
