// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       modelica.raw_function @scalarVariable() -> !modelica.int {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[result:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      modelica.raw_return %[[result]]
// CHECK-NEXT:  }

modelica.function @scalarVariable {
    modelica.variable @x : !modelica.variable<!modelica.int, output>
}

// -----

// Get a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableGet() -> !modelica.int {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      modelica.print %[[x_value]]
// CHECK:       }

modelica.function @scalarVariableGet {
    modelica.variable @x : !modelica.variable<!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.int
        modelica.print %1 : !modelica.int
    }
}

// -----

// Set a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableSet() -> !modelica.int {
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

modelica.function @scalarVariableSet {
    modelica.variable @x : !modelica.variable<!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.constant #modelica.int<0>
        modelica.variable_set @x, %1 : !modelica.int
    }
}

// -----

// Static array.

// CHECK:       modelica.raw_function @staticArray() -> !modelica.array<3x2x!modelica.int> {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x2x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      modelica.raw_return %[[x_value]]
// CHECK-NEXT:  }

modelica.function @staticArray {
    modelica.variable @x : !modelica.variable<3x2x!modelica.int, output>
}

// -----

// Get a static array.

// CHECK:       modelica.raw_function @staticArrayGet() -> !modelica.array<3x2x!modelica.int> {
// CHECK:           %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x2x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = modelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK:       }

modelica.function @staticArrayGet {
    modelica.variable @x : !modelica.variable<3x2x!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x2x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Set a static array.

// CHECK:   modelica.raw_function @staticArraySet() -> !modelica.array<3x2x!modelica.int> {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x2x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = modelica.alloc
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:   }

modelica.function @staticArraySet {
    modelica.variable @x : !modelica.variable<3x2x!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
        modelica.variable_set @x, %1 : !modelica.array<3x2x!modelica.int>
    }
}

// -----

// Dynamic array.

// CHECK:       modelica.raw_function @dynamicArray() -> !modelica.array<3x?x!modelica.int> {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[result:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      modelica.raw_return %[[result]]
// CHECK-NEXT:  }

modelica.function @dynamicArray {
    modelica.variable @x : !modelica.variable<3x?x!modelica.int, output>
}

// -----

// Get a dynamic array.

// CHECK:       modelica.raw_function @dynamicArrayGet() -> !modelica.array<3x?x!modelica.int> {
// CHECK:           %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = modelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]] : !modelica.int
// CHECK:       }

modelica.function @dynamicArrayGet {
    modelica.variable @x : !modelica.variable<3x?x!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x?x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Set a dynamic array.

// CHECK:       modelica.raw_function @dynamicArraySet() -> !modelica.array<3x?x!modelica.int> {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = modelica.alloc
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:   }

modelica.function @dynamicArraySet {
    modelica.variable @x : !modelica.variable<3x?x!modelica.int, output>

    modelica.algorithm {
        %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
        modelica.variable_set @x, %1 : !modelica.array<3x2x!modelica.int>
    }
}

// -----

// Scalar default value.

// CHECK:       modelica.raw_function @scalarDefaultValue() -> !modelica.int {
// CHECK-DAG:       %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[non_default:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
// CHECK:           modelica.raw_variable_set %[[x]], %[[default]]
// CHECK:           modelica.raw_variable_set %[[x]], %[[non_default]]
// CHECK:       }

modelica.function @scalarDefaultValue {
    modelica.variable @x : !modelica.variable<!modelica.int, output>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.constant #modelica.int<1>
        modelica.variable_set @x, %0 : !modelica.int
    }
}

// -----

// Array default value.

// CHECK:       modelica.raw_function @arrayDefaultValue() -> !modelica.array<3x!modelica.int> {
// CHECK-DAG:       %[[default:.*]] = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
// CHECK-DAG:       %[[non_default:.*]] = modelica.constant #modelica.int_array<[1, 1, 1]> : !modelica.array<3x!modelica.int>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.variable<3x!modelica.int, output> {name = "x"}
// CHECK:           modelica.raw_variable_set %[[x]], %[[default]]
// CHECK:           modelica.raw_variable_set %[[x]], %[[non_default]]
// CHECK:       }

modelica.function @arrayDefaultValue {
    modelica.variable @x : !modelica.variable<3x!modelica.int, output>

    modelica.default @x {
        %0 = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
        modelica.yield %0 : !modelica.array<3x!modelica.int>
    }

    modelica.algorithm {
        %0 = modelica.constant #modelica.int_array<[1, 1, 1]> : !modelica.array<3x!modelica.int>
        modelica.variable_set @x, %0 : !modelica.array<3x!modelica.int>
    }
}
