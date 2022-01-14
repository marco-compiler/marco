// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 57

#include <array>
#include <iostream>

extern "C" long __modelica_ciface_foo();

using namespace std;

int main() {
	cout << "results" << endl;
	cout << __modelica_ciface_foo() << endl;

	return 0;
}
