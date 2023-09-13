// RUN: modelica-opt %s --split-input-file --allocate-derivatives --canonicalize | FileCheck %s

// Check variable declaration and derivatives map.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,4]}>
// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x, #[[index_set]]>]
// CHECK-DAG: modelica.variable @x : !modelica.variable<5x!modelica.real>
// CHECK-DAG: modelica.variable @der_x : !modelica.variable<5x!modelica.real>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        modelica.store %0[%1], %3 : !modelica.array<5x!modelica.real>
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       modelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = modelica.constant 3 : index
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-NEXT:      %[[load:.*]] = modelica.load %[[der_x]][%[[index]]]
// CHECK-NEXT:      modelica.store %[[x]][%[[index]]], %[[load]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        modelica.store %0[%1], %3 : !modelica.array<5x!modelica.real>
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[zero]] : !modelica.real -> !modelica.array<5x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        modelica.store %0[%1], %3 : !modelica.array<5x!modelica.real>
    }
}

// -----

// Check equations for non-derived indices.

// CHECK-NOT:   modelica.equation_template

modelica.model @Test {
    modelica.variable @x : !modelica.variable<5x!modelica.real>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.array<5x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.load %0[%1] : !modelica.array<5x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        modelica.store %0[%1], %3 : !modelica.array<5x!modelica.real>
    }
}
