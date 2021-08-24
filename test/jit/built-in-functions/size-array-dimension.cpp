// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 3

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(ArrayDescriptor<long, 2>* x);

using namespace std;

int main() {
	array<long, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 3 });

	cout << "results" << endl;
	cout << __modelica_ciface_foo(&xDescriptor)<< endl;

	return 0;
}
