// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[37, 54], [81, 118]]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* x, long y, ArrayDescriptor<long, 2>* z);

using namespace std;

int main() {
	array<long, 4> x = { 1, 2, 3, 4 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 2 });

	long y = 3;

	array<long, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<long, 2> zDescriptor(z.data(), { 2, 2 });

	__modelica_ciface_foo(&xDescriptor, y, &zDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
