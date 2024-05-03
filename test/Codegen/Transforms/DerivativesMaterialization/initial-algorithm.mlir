// RUN: modelica-opt %s --split-input-file --derivatives-materialization --canonicalize | FileCheck %s

// Check variable declaration and derivatives map.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,4]}>
// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#bmodelica<var_derivative @x, @der_x, #[[index_set]]>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<5x!bmodelica.real>

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.algorithm attributes {initial = true} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        bmodelica.store %0[%1], %3 : !bmodelica.array<5x!bmodelica.real>
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.algorithm attributes {initial = true} {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[load:.*]] = bmodelica.load %[[der_x]][%[[index]]]
// CHECK-NEXT:      bmodelica.store %[[x]][%[[index]]], %[[load]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.algorithm attributes {initial = true} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        bmodelica.store %0[%1], %3 : !bmodelica.array<5x!bmodelica.real>
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK-NEXT:      %[[array:.*]] = bmodelica.array_broadcast %[[zero]] : !bmodelica.real -> <5x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[array]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.algorithm attributes {initial = true} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        bmodelica.store %0[%1], %3 : !bmodelica.array<5x!bmodelica.real>
    }
}

// -----

// Check equations for non-derived indices.

// CHECK-NOT:   bmodelica.equation_template

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.algorithm attributes {initial = true} {
        %0 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        bmodelica.store %0[%1], %3 : !bmodelica.array<5x!bmodelica.real>
    }
}
