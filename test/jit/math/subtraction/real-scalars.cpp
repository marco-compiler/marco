// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: -1.000000e+00
// CHECK-NEXT: 1.000000e+00
// CHECK-NEXT: 4.000000e+00
// CHECK-NEXT: -4.000000e+00

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>

extern "C" double __modelica_ciface_foo(double x, double y);

using namespace std;

int main() {
	array<double, 4> x = { 1.5, -1.5, 1.5, -1.5 };
	array<double, 4> y = { 2.5, -2.5, -2.5, 2.5 };

	cout << "results" << endl;

	for (const auto& [x, y] : llvm::zip(x, y))
		cout << scientific << __modelica_ciface_foo(x, y) << endl;

	return 0;
}