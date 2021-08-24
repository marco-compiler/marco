// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 7.853982e-01
// CHECK-NEXT: 2.356194e+00
// CHECK-NEXT: -2.356194e+00
// CHECK-NEXT: -7.853982e-01
// CHECK-NEXT: 0.000000e+00

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" double __modelica_ciface_foo(double y, double x);

using namespace std;

int main() {
	array<double, 5> y = { 0.707106781, 0.707106781, -0.707106781, -0.707106781 };
	array<double, 5> x = { 0.707106781, -0.707106781, -0.707106781, 0.707106781 };
	cout << "results" << endl;

	for (const auto& [y, x] : llvm::zip(y, x))
		cout << scientific << __modelica_ciface_foo(y, x) << endl;

	return 0;
}
