// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       modelica.raw_function @scalarVariable() {
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariable {
    modelica.variable @x : !modelica.member<!modelica.int>
}

// -----

// Get a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableGet() {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.member<!modelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      modelica.print %[[x_value]]
// CHECK:       }

modelica.function @scalarVariableGet {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.int
        modelica.print %1 : !modelica.int
    }
}

// -----

// Set a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableSet() {
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.member<!modelica.int> {name = "x"}
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

modelica.function @scalarVariableSet {
    modelica.variable @x : !modelica.member<!modelica.int>

    modelica.algorithm {
        %1 = modelica.constant #modelica.int<0>
        modelica.variable_set @x, %1 : !modelica.int
    }
}

// -----

// Static array.

// CHECK:       modelica.raw_function @staticArray() {
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArray {
    modelica.variable @x : !modelica.member<3x2x!modelica.int>
}

// -----

// Get a static array.

// CHECK:       modelica.raw_function @staticArrayGet() {
// CHECK:           %[[x:.*]] = modelica.raw_variable : !modelica.member<3x2x!modelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = modelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK:       }

modelica.function @staticArrayGet {
    modelica.variable @x : !modelica.member<3x2x!modelica.int>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x2x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Set a static array.

// CHECK:       modelica.raw_function @staticArraySet() {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.member<3x2x!modelica.int> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = modelica.alloc
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

modelica.function @staticArraySet {
    modelica.variable @x : !modelica.member<3x2x!modelica.int>

    modelica.algorithm {
        %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
        modelica.variable_set @x, %1 : !modelica.array<3x2x!modelica.int>
    }
}

// -----

// Dynamic array.

// CHECK:       modelica.raw_function @dynamicArray() {
// CHECK:           modelica.raw_return
// CHECK-NEXT:  }

modelica.function @dynamicArray {
    modelica.variable @x : !modelica.member<3x?x!modelica.int>
}

// -----

// Get a dynamic array.

// CHECK:       modelica.raw_function @dynamicArrayGet() {
// CHECK:           %[[x:.*]] = modelica.raw_variable : !modelica.member<3x?x!modelica.int> {name = "x"}
// CHECK-NEXT:      %[[x_value:.*]] = modelica.raw_variable_get %[[x]]
// CHECK-NEXT:      %[[value:.*]] = modelica.load %[[x_value]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK:       }

modelica.function @dynamicArrayGet {
    modelica.variable @x : !modelica.member<3x?x!modelica.int>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x?x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Set a dynamic array.

// CHECK:       modelica.raw_function @dynamicArraySet() {
// CHECK-NEXT:      %[[x:.*]] = modelica.raw_variable : !modelica.member<3x?x!modelica.int> {name = "x"}
// CHECK-NEXT:      %[[value:.*]] = modelica.alloc
// CHECK-NEXT:      modelica.raw_variable_set %[[x]], %[[value]]
// CHECK:       }

modelica.function @dynamicArraySet {
    modelica.variable @x : !modelica.member<3x?x!modelica.int>

    modelica.algorithm {
        %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
        modelica.variable_set @x, %1 : !modelica.array<3x2x!modelica.int>
    }
}

// -----

// Scalar default value.

// CHECK:       modelica.raw_function @scalarDefaultValue() {
// CHECK-DAG:       %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[non_default:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.member<!modelica.int> {name = "x"}
// CHECK:           modelica.raw_variable_set %[[x]], %[[default]]
// CHECK:           modelica.raw_variable_set %[[x]], %[[non_default]]
// CHECK:       }

modelica.function @scalarDefaultValue {
    modelica.variable @x : !modelica.member<!modelica.int>

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

// CHECK:       modelica.raw_function @arrayDefaultValue() {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[one:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG:       %[[x:.*]] = modelica.raw_variable : !modelica.member<3x!modelica.int> {name = "x"}
// CHECK:           %[[default:.*]] = modelica.array_broadcast %[[zero]]
// CHECK:           modelica.raw_variable_set %[[x]], %[[default]]
// CHECK:           %[[non_default:.*]] = modelica.array_broadcast %[[one]]
// CHECK:           modelica.raw_variable_set %[[x]], %[[non_default]]
// CHECK:       }

modelica.function @arrayDefaultValue {
    modelica.variable @x : !modelica.member<3x!modelica.int>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.yield %1 : !modelica.array<3x!modelica.int>
    }

    modelica.algorithm {
        %0 = modelica.constant #modelica.int<1>
        %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x!modelica.int>
        modelica.variable_set @x, %1 : !modelica.array<3x!modelica.int>
    }
}
