// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: true
// CHECK-NEXT: true

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>

extern "C" bool __modelica_ciface_foo(long x, double y);

using namespace std;

int main() {
	array<long, 5> x = { 0, 10, 10, 10, -10 };
	array<double, 5> y = { 0.5, 10.5, 10, 10.5, -9.5 };

	cout << "results" << endl;

	for (const auto& [x, y] : llvm::zip(x, y))
		cout << boolalpha << __modelica_ciface_foo(x, y) << endl;

	return 0;
}
