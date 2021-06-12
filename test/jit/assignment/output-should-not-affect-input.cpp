// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [1, 2]
// CHECK-NEXT{LITERAL}: [10, 2]

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* y, ArrayDescriptor<long, 1>* x);

using namespace std;

int main() {
	array<long, 2> x = { 1, 2 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	ArrayDescriptor<long, 1> yDescriptor(nullptr, { 1 });

	cout << "results" << endl;
	__modelica_ciface_foo(&yDescriptor, &xDescriptor);
	cout << xDescriptor << endl;
	cout << yDescriptor << endl;
	free(yDescriptor.getData());

	return 0;
}
