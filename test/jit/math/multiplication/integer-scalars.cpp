// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 0
// CHECK-NEXT: 15
// CHECK-NEXT: -15
// CHECK-NEXT: -15

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>

extern "C" long __modelica_ciface_foo(long x, long y);

using namespace std;

int main() {
	array<long, 4> x = { 0, 3, 3, -5 };
	array<long, 4> y = { 10, 5, -5, 3 };

	cout << "results" << endl;

	for (const auto& [x, y] : llvm::zip(x, y))
		cout << __modelica_ciface_foo(x, y) << endl;

	return 0;
}
