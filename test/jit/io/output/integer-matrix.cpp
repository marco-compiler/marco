// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4
// CHECK-NEXT: 5
// CHECK-NEXT: 6

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* x);

using namespace std;

int main() {
	array<long, 6> x = { 0, 0, 0, 0, 0, 0 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 3 });

	__modelica_ciface_foo(&xDescriptor);

	cout << "results" << endl;

	for (size_t i = 0; i < 2; ++i)
		for (size_t j = 0; j < 3; ++j)
			cout << xDescriptor.get(i, j) << endl;

	return 0;
}
