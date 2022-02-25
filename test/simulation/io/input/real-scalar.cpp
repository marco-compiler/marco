// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mod
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
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