// RUN: marco %p/neg-scalar-integer.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ -g -c %s -I %runtime_h -I %llvm_include_dirs $(llvm-config --cxxflags) -o %basename_t.o
// RUN: clang++ -g %basename_t.o %basename_t.mo.s $(llvm-config --ldflags --libs) -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: -23
// CHECK-NEXT: 57

#include <array>
#include <iostream>

extern "C" long __modelica_ciface_negScalarInteger(long x);

using namespace std;

int main() {
	array<long, 2> x = { 23, -57 };

	cout << "results" << endl;

	for (const auto& value : x)
		cout << __modelica_ciface_negScalarInteger(value) << endl;

	return 0;
}
