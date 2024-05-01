// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-bmodelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariable(%{{.*}}: !bmodelica.int) {
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
}

// -----

// Get a scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariableGet(%[[x:.*]]: !bmodelica.int) {
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @scalarVariableGet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %1 : !bmodelica.int
    }
}

// -----

// Static array.

// CHECK:       bmodelica.raw_function @staticArray(%{{.*}}: !bmodelica.array<3x2x!bmodelica.int>) {
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @staticArray {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, input>
}

// -----

// Get a static array.

// CHECK:       bmodelica.raw_function @staticArrayGet(%[[x:.*]]: !bmodelica.array<3x2x!bmodelica.int>) {
// CHECK:           %[[value:.*]] = bmodelica.load %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @staticArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, input>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x2x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x2x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}

// -----

// Dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArray(%{{.*}}: !bmodelica.array<3x?x!bmodelica.int>) {
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @dynamicArray {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, input>
}

// -----

// Get a dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArrayGet(%[[x:.*]]: !bmodelica.array<3x?x!bmodelica.int>) {
// CHECK-NEXT:      %[[index:.*]]= arith.constant 0 : index
// CHECK-NEXT:      %[[value:.*]] = bmodelica.load %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @dynamicArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, input>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x?x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x?x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}
