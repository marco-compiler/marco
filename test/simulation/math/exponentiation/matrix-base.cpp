// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[37, 54], [81, 118]]

#include <array>
#include <iostream>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* x, long y, ArrayDescriptor<long, 2>* z);

using namespace std;

int main() {
	array<long, 4> x = { 1, 2, 3, 4 };
	ArrayDescriptor<long, 2> xDescriptor(x.data(), { 2, 2 });

	long y = 3;

	array<long, 4> z = { 0, 0, 0, 0 };
	ArrayDescriptor<long, 2> zDescriptor(z.data(), { 2, 2 });

	__modelica_ciface_foo(&xDescriptor, y, &zDescriptor);

	cout << "results" << endl;
	cout << zDescriptor << endl;

	return 0;
}
