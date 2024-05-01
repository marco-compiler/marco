// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-bmodelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariable() {
// CHECK:           bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
}

// -----

// Get a scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariableGet() {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      bmodelica.print %[[x_value]]
// CHECK:       }

bmodelica.function @scalarVariableGet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %1 : !bmodelica.int
    }
}

// -----

// Set a scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariableSet() {
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int> {name = "x"}
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

bmodelica.function @scalarVariableSet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.constant #bmodelica.int<0>
        bmodelica.variable_set @x, %1 : !bmodelica.int
    }
}

// -----

// Static array.

// CHECK:       bmodelica.raw_function @staticArray() {
// CHECK:           bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @staticArray {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int>
}

// -----

// Get a static array.

// CHECK:       bmodelica.raw_function @staticArrayGet() {
// CHECK:           %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x2x!bmodelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = bmodelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK:       }

bmodelica.function @staticArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x2x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x2x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}

// -----

// Set a static array.

// CHECK:       bmodelica.raw_function @staticArraySet() {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x2x!bmodelica.int> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = bmodelica.alloc
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

bmodelica.function @staticArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.alloc : <3x2x!bmodelica.int>
        bmodelica.variable_set @x, %1 : !bmodelica.array<3x2x!bmodelica.int>
    }
}

// -----

// Dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArray() {
// CHECK:           bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @dynamicArray {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int>
}

// -----

// Get a dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArrayGet() {
// CHECK:           %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = bmodelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK:       }

bmodelica.function @dynamicArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x?x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x?x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}

// -----

// Set a dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArraySet() {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = bmodelica.alloc
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

bmodelica.function @dynamicArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int>

    bmodelica.algorithm {
        %1 = bmodelica.alloc : <3x2x!bmodelica.int>
        bmodelica.variable_set @x, %1 : !bmodelica.array<3x2x!bmodelica.int>
    }
}
