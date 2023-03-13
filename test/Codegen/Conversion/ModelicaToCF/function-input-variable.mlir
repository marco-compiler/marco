// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false}, canonicalize, cse)" | FileCheck %s

// Scalar variable.

// CHECK:       modelica.raw_function @scalarVariable(%{{.*}}: !modelica.int) {
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariable {
    modelica.variable @x : !modelica.member<!modelica.int, input>
}

// -----

// Get a scalar variable.

// CHECK:       modelica.raw_function @scalarVariableGet(%[[x:.*]]: !modelica.int) {
// CHECK-NEXT:      modelica.print %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariableGet {
    modelica.variable @x : !modelica.member<!modelica.int, input>

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
    modelica.variable @x : !modelica.member<3x2x!modelica.int, input>
}

// -----

// Get a static array.

// CHECK:       modelica.raw_function @staticArrayGet(%[[x:.*]]: !modelica.array<3x2x!modelica.int>) {
// CHECK:           %[[value:.*]] = modelica.load %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      modelica.print %[[value]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArrayGet {
    modelica.variable @x : !modelica.member<3x2x!modelica.int, input>

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
    modelica.variable @x : !modelica.member<3x?x!modelica.int, input>
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
    modelica.variable @x : !modelica.member<3x?x!modelica.int, input>

    modelica.algorithm {
        %1 = modelica.variable_get @x : !modelica.array<3x?x!modelica.int>
        %2 = arith.constant 0 : index
        %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
        modelica.print %3 : !modelica.int
    }
}
