// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       modelica.raw_function @scalarVariable(%{{.*}}: !modelica.int) {
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariable {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
}

// -----

// Get a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableGet(%[[x:.*]]: !modelica.int) {
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariableGet {
    modelica.variable @x : !modelica.variable<!modelica.int, input>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.int
        modelica.print %1 : !modelica.int
    }
}

// -----

// Static array.

// CHECK:       modelica.raw_function @staticArray(%{{.*}}: !modelica.array<3x2x!modelica.int>) {
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArray {
    modelica.variable @x : !modelica.variable<3x2x!modelica.int, input>
}

// -----

// Get a static array.

// CHECK:       modelica.raw_function @staticArrayGet(%[[x:.*]]: !modelica.array<3x2x!modelica.int>) {
// CHECK:           %[[value:.*]] = modelica.load %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArrayGet {
    modelica.variable @x : !modelica.variable<3x2x!modelica.int, input>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x2x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Dynamic array.

// CHECK:       modelica.raw_function @dynamicArray(%{{.*}}: !modelica.array<3x?x!modelica.int>) {
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @dynamicArray {
    modelica.variable @x : !modelica.variable<3x?x!modelica.int, input>
}

// -----

// Get a dynamic array.

// CHECK:       modelica.raw_function @dynamicArrayGet(%[[x:.*]]: !modelica.array<3x?x!modelica.int>) {
// CHECK-NEXT:      %[[index:.*]]= arith.constant 0 : index
// CHECK-NEXT:      %[[value:.*]] = modelica.load %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @dynamicArrayGet {
    modelica.variable @x : !modelica.variable<3x?x!modelica.int, input>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x?x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}

// -----

// Scalar default value.

// CHECK-LABEL: @caller
// CHECK: %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK: modelica.call @scalarDefaultValue(%[[default]]) : (!modelica.int) -> ()

modelica.function @scalarDefaultValue {
    modelica.variable @x : !modelica.variable<!modelica.int, input>

    modelica.default @x {
        %0 = modelica.constant #modelica.int<0>
        modelica.yield %0 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        modelica.print %0 : !modelica.int
    }
}

func.func @caller() {
    modelica.call @scalarDefaultValue() : () -> ()
    func.return
}

// -----

// Array default value.

// CHECK-LABEL: @caller
// CHECK: %[[cst:.*]] = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
// CHECK: modelica.call @arrayDefaultValue(%[[cst]]) : (!modelica.array<3x!modelica.int>) -> ()

modelica.function @arrayDefaultValue {
    modelica.variable @x : !modelica.variable<3x!modelica.int, input>

    modelica.default @x {
        %0 = modelica.constant #modelica.int_array<[0, 0, 0]> : !modelica.array<3x!modelica.int>
        modelica.yield %0 : !modelica.array<3x!modelica.int>
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<3x!modelica.int>
        modelica.print %0 : !modelica.array<3x!modelica.int>
    }
}

func.func @caller() {
    modelica.call @arrayDefaultValue() : () -> ()
    func.return
}

// -----

// Not all arguments passed.

// CHECK-LABEL: @caller
// CHECK-DAG: %[[x:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG: %[[y:.*]] = modelica.constant #modelica.int<1>
// CHECK-DAG: %[[z:.*]] = modelica.constant #modelica.int<2>
// CHECK: modelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!modelica.int, !modelica.int, !modelica.int) -> ()

modelica.function @missingArguments {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
    modelica.variable @y : !modelica.variable<!modelica.int, input>
    modelica.variable @z : !modelica.variable<!modelica.int, input>

    modelica.default @y {
        %0 = modelica.constant #modelica.int<1>
        modelica.yield %0 : !modelica.int
    }

    modelica.default @z {
        %0 = modelica.constant #modelica.int<2>
        modelica.yield %0 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.variable_get @z : !modelica.int
        modelica.print %0 : !modelica.int
        modelica.print %1 : !modelica.int
        modelica.print %2 : !modelica.int
    }
}

func.func @caller() {
    %0 = modelica.constant #modelica.int<0>
    modelica.call @missingArguments(%0) : (!modelica.int) -> ()
    func.return
}

// -----

// Missing arguments with dependencies.

// CHECK-LABEL: @caller
// CHECK-SAME:  (%[[x:.*]]: !modelica.int)
// CHECK-DAG: %[[two:.*]] = modelica.constant #modelica.int<2>
// CHECK-DAG: %[[three:.*]] = modelica.constant #modelica.int<3>
// CHECK-DAG: %[[z:.*]] = modelica.mul %[[x]], %[[two]]
// CHECK-DAG: %[[y:.*]] = modelica.mul %[[z]], %[[three]]
// CHECK: modelica.call @missingArguments(%[[x]], %[[y]], %[[z]]) : (!modelica.int, !modelica.int, !modelica.int) -> ()

modelica.function @missingArguments {
    modelica.variable @x : !modelica.variable<!modelica.int, input>
    modelica.variable @y : !modelica.variable<!modelica.int, input>
    modelica.variable @z : !modelica.variable<!modelica.int, input>

    modelica.default @y {
        %0 = modelica.variable_get @z : !modelica.int
        %1 = modelica.constant #modelica.int<3>
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %2 : !modelica.int
    }

    modelica.default @z {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.constant #modelica.int<2>
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %2 : !modelica.int
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        %1 = modelica.variable_get @y : !modelica.int
        %2 = modelica.variable_get @z : !modelica.int
        modelica.print %0 : !modelica.int
        modelica.print %1 : !modelica.int
        modelica.print %2 : !modelica.int
    }
}

func.func @caller(%arg0: !modelica.int) {
    modelica.call @missingArguments(%arg0) : (!modelica.int) -> ()
    func.return
}

