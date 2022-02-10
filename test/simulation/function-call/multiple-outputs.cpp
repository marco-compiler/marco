// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mod
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 2
// CHECK-NEXT: 3

#include <array>
#include <iostream>

struct Result {
	long y;
	long z;
};

extern "C" void __modelica_ciface_foo(Result* result, long x);

using namespace std;

int main() {
	long x = 1;
	Result result;

	__modelica_ciface_foo(&result, x);

	cout << "results" << endl;
	cout << result.y << endl;
	cout << result.z << endl;

	return 0;
}
