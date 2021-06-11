// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 2
// CHECK-NEXT: 2
// CHECK-NEXT: -1
// CHECK-NEXT: -1
// CHECK-NEXT: 1
// CHECK-NEXT: 1

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>

extern "C" long __modelica_ciface_foo(long x, long y);

using namespace std;

int main() {
	array<long, 6> x = { 1, 2, -1, -2, 1, -1 };
	array<long, 6> y = { 2, 1, -2, -1, -1, 1 };

	cout << "results" << endl;

	for (const auto& [x, y] : llvm::zip(x, y))
		cout << __modelica_ciface_foo(x, y) << endl;

	return 0;
}
