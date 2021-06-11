// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [-1, 1, 3, -3]

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(
		ArrayDescriptor<long, 1>* z, ArrayDescriptor<long, 1>* x, ArrayDescriptor<long, 1>* y);

using namespace std;

int main() {
	array<long, 4> x = { 1, -1, 1, -1 };
	ArrayDescriptor<long, 1> xDescriptor(x);

	array<long, 4> y = { 2, -2, -2, 2 };
	ArrayDescriptor<long, 1> yDescriptor(y);

	ArrayDescriptor<long, 1> zDescriptor(nullptr, { 1 });

	__modelica_ciface_foo(&zDescriptor, &xDescriptor, &yDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;
	free(zDescriptor.getData());

	return 0;
}
