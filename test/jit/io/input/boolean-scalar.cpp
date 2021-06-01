// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) -o %t
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
