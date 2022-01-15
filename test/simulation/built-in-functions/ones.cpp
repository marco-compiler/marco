// RUN: marco --omc-bypass -c -O0 -c-wrappers -o %basename_t.mo %s.mo
// RUN: g++ %s -c -std=c++1z -I %marco_include_dirs -I %llvm_include_dirs -o %basename_t.o
// RUN: g++ %basename_t.o %basename_t.mo.o %runtime_lib $(llvm-config --ldflags --libs) -Wl,-R%libs -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [1, 1, 1]]
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1], [1, 1]]

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>
#include <marco/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<long, 2>* y, long n1, long n2);

using namespace std;

int main() {
	array<long, 2> n1 = { 2, 3 };
	array<long, 2> n2 = { 3, 2 };

	ArrayDescriptor<long, 2> yDescriptor(nullptr, { 1, 1 });

	cout << "results" << endl;

	for (const auto& [n1, n2] : llvm::zip(n1, n2))
	{
		__modelica_ciface_foo(&yDescriptor, n1, n2);
		cout << yDescriptor << endl;
		free(yDescriptor.getData());
	}

	return 0;
}