// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: -3

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(ArrayDescriptor<long, 2>* x);

using namespace std;

int main() {
	array<long, 6> x = { -1, 9, -3, 0, 4, 3 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 3, 2 });

	cout << "results" << endl;
	cout << __modelica_ciface_foo(&xDescriptor)<< endl;

	return 0;
}
