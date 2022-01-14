// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 5
// CHECK-NEXT: 81
// CHECK-NEXT: 1
// CHECK-NEXT: 16
// CHECK-NEXT: 0
// CHECK-NEXT: -8
// CHECK-NEXT: 4

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>

extern "C" long __modelica_ciface_foo(long x, long y);

using namespace std;

int main() {
	array<int, 7> x = { 5, 3, 2, 4, 0, -2, -2 };
	array<int, 7> y = { 1, 4, 0, 2, 3, 3, 2 };

	cout << "results" << endl;

	for (const auto& [x, y] : llvm::zip(x, y))
		cout << __modelica_ciface_foo(x, y) << endl;

	return 0;
}
