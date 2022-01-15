// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT: 6
// CHECK-NEXT: 15

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

struct Result {
	long y = 0;
	long z = 0;
};

extern "C" void __modelica_ciface_foo(Result* result, ArrayDescriptor<long, 2>* x);

using namespace std;

int main() {
	array<long, 6> x = { 1, 2, 3, 4, 5, 6 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 3 });
	Result result;

	__modelica_ciface_foo(&result, &xDescriptor);

	cout << "results" << endl;
	cout << result.y << endl;
	cout << result.z << endl;

	return 0;
}