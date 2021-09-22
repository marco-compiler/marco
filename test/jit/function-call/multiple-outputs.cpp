// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g $(llvm-config --ldflags --libs) %runtime_lib -Wl,-R%libs/runtime -o %t
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
