// RUN: marco %s.mo --emit-c-wrappers | llc -o %basename_t.mo.s
// RUN: clang++ %s -g -c -std=c++1z -I %runtime_h -I %llvm_include_dirs -o %basename_t.o
// RUN: clang++ %basename_t.o %basename_t.mo.s -g -L%libs/runtime $(llvm-config --ldflags --libs) -lruntime -Wl,-R%libs/runtime -o %t
// RUN: %t | FileCheck %s

// CHECK-LABEL: results
// CHECK-NEXT{LITERAL}: [[1]]
// CHECK-NEXT{LITERAL}: [[1, 0], [0, 1]]
// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

#include <array>
#include <iostream>
#include <modelica/runtime/ArrayDescriptor.h>

extern "C" long __modelica_ciface_foo(ArrayDescriptor<long, 2>* y, long x);

using namespace std;

int main() {
	array<long, 3> x = { 1, 2, 3 };
	ArrayDescriptor<long, 2> yDescriptor(nullptr, { 1, 1 });

	cout << "results" << endl;

	for (const auto& value : x)
	{
		__modelica_ciface_foo(&yDescriptor, value);

		cout << "[";

		for (size_t i = 0; i < yDescriptor.getDimensionSize(0); ++i)
		{
			if (i > 0)
				cout << ", ";

			cout << "[";

			for (size_t j = 0; j < yDescriptor.getDimensionSize(1); ++j)
			{
				if (j > 0)
					cout << ", ";

				cout << yDescriptor.get(i, j);
			}

			cout << "]";
		}

		cout << "]" << endl;
		delete[] yDescriptor.getData();
	}

	return 0;
}
