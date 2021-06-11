// RUN: marco %s.mo --O0 --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[1, 1, 1], [1, 1, 1]]
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1], [1, 1]]

#include <array>
#include <iostream>
#include <llvm/ADT/STLExtras.h>
#include <modelica/runtime/ArrayDescriptor.h>

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
