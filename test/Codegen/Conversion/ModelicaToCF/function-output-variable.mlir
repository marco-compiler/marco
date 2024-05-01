// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariable() -> !bmodelica.int {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      bmodelica.raw_return %[[result]]
// CHECK-NEXT:  }

bmodelica.function @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>
}

// -----

// Get a scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariableGet() -> !bmodelica.int {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      bmodelica.print %[[x_value]]
// CHECK:       }

bmodelica.function @scalarVariableGet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %1 : !bmodelica.int
    }
}

// -----

// Set a scalar variable.

// CHECK:       bmodelica.raw_function @scalarVariableSet() -> !bmodelica.int {
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

bmodelica.function @scalarVariableSet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.constant #bmodelica.int<0>
        bmodelica.variable_set @x, %1 : !bmodelica.int
    }
}

// -----

// Static array.

// CHECK:       bmodelica.raw_function @staticArray() -> !bmodelica.array<3x2x!bmodelica.int> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x2x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      bmodelica.raw_return %[[x_value]]
// CHECK-NEXT:  }

bmodelica.function @staticArray {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, output>
}

// -----

// Get a static array.

// CHECK:       bmodelica.raw_function @staticArrayGet() -> !bmodelica.array<3x2x!bmodelica.int> {
// CHECK:           %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x2x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = bmodelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK:       }

bmodelica.function @staticArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x2x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x2x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}

// -----

// Set a static array.

// CHECK:   bmodelica.raw_function @staticArraySet() -> !bmodelica.array<3x2x!bmodelica.int> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x2x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = bmodelica.alloc
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:   }

bmodelica.function @staticArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.alloc : <3x2x!bmodelica.int>
        bmodelica.variable_set @x, %1 : !bmodelica.array<3x2x!bmodelica.int>
    }
}

// -----

// Dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArray() -> !bmodelica.array<3x?x!bmodelica.int> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[result:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      bmodelica.raw_return %[[result]]
// CHECK-NEXT:  }

bmodelica.function @dynamicArray {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, output>
}

// -----

// Get a dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArrayGet() -> !bmodelica.array<3x?x!bmodelica.int> {
// CHECK:           %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = bmodelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = bmodelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]] : !bmodelica.int
// CHECK:       }

bmodelica.function @dynamicArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.variable_get @x : !bmodelica.array<3x?x!bmodelica.int>
        %2 = arith.constant 0 : index
        %3 = bmodelica.load %1[%2, %2] : !bmodelica.array<3x?x!bmodelica.int>
        bmodelica.print %3 : !bmodelica.int
    }
}

// -----

// Set a dynamic array.

// CHECK:       bmodelica.raw_function @dynamicArraySet() -> !bmodelica.array<3x?x!bmodelica.int> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = bmodelica.alloc
// CHECK-NEXT:      bmodelica.raw_variable_set %[[x]], %[[value]]
// CHECK:   }

bmodelica.function @dynamicArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, output>

    bmodelica.algorithm {
        %1 = bmodelica.alloc : <3x2x!bmodelica.int>
        bmodelica.variable_set @x, %1 : !bmodelica.array<3x2x!bmodelica.int>
    }
}
