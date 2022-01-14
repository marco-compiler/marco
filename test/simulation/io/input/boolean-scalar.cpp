// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: false
// CHECK-NEXT: true

#include <array>
#include <iostream>

extern "C" bool __modelica_ciface_foo(bool x);

using namespace std;

int main() {
	array<bool, 2> x = { false, true };

	cout << "results" << endl;

	for (const auto& value : x)
		cout << boolalpha << __modelica_ciface_foo(value) << endl;

	return 0;
}
