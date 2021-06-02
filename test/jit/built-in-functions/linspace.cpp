// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01, 9.000000e-01, 1.000000e+00]
// CHECK-NEXT{LITERAL}: [-1.000000e+00, -9.000000e-01, -8.000000e-01, -7.000000e-01, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, -2.000000e-01, -1.000000e-01, 0.000000e+00]

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" void __modelica_ciface_foo(ArrayDescriptor<double, 1>* y, long start, long stop, long n);

using namespace std;

int main() {
	array<long, 2> start = { 0, -1 };
	array<long, 2> stop = { 1, 0 };
	array<long, 2> n = { 11, 11 };

	ArrayDescriptor<double, 1> yDescriptor(nullptr, { 1 });

	cout << "results" << endl;

	for (const auto& [start, stop, n] : llvm::zip(start, stop, n))
	{
		__modelica_ciface_foo(&yDescriptor, start, stop, n);
		cout << yDescriptor << endl;
		free(yDescriptor.getData());
	}

	return 0;
}
