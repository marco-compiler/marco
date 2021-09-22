// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 0
// CHECK-NEXT: 10.5
// CHECK-NEXT: -10.5

#include <array>
#include <iostream>

extern "C" double __modelica_ciface_foo(double x);

using namespace std;

int main() {
	array<double, 3> x = { 0, 10.5, -10.5 };

	cout << "results" << endl;

	for (const auto& value : x)
		cout << __modelica_ciface_foo(value) << endl;

	return 0;
}
