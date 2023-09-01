// RUN: modelica-opt %s --split-input-file --allocate-derivatives --canonicalize | FileCheck %s

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x, {[3,5][12,14]}>]
// CHECK-DAG: modelica.variable @x : !modelica.variable<10x20x!modelica.real>
// CHECK-DAG: modelica.variable @der_x : !modelica.variable<10x20x!modelica.real>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x20x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x20x!modelica.real>
        %1 = modelica.load %0[%i0, %i1] : !modelica.array<10x20x!modelica.real>
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        %3 = modelica.constant #modelica.real<3.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>} : !modelica.equation
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
// CHECK:           %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:           %[[load:.*]] = modelica.load %[[der_x]][%[[i0]], %[[i1]]]
// CHECK:           %[[lhs:.*]] = modelica.equation_side %[[load]]
// CHECK:           modelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][12,14]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x20x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x20x!modelica.real>
        %1 = modelica.load %0[%i0, %i1] : !modelica.array<10x20x!modelica.real>
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        %3 = modelica.constant #modelica.real<3.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>} : !modelica.equation
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[zero]] : !modelica.real -> !modelica.array<10x20x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x20x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x20x!modelica.real>
        %1 = modelica.load %0[%i0, %i1] : !modelica.array<10x20x!modelica.real>
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        %3 = modelica.constant #modelica.real<3.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>} : !modelica.equation
}

// -----

// Check equations for non-derived indices.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-DAG:       %[[load:.*]] = modelica.load %[[der_x]][%[[i0]], %[[i1]]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2][0,19]>}
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][0,11]>}
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][15,19]>}
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [6,9][0,19]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x20x!modelica.real>

    %t0 = modelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x20x!modelica.real>
        %1 = modelica.load %0[%i0, %i1] : !modelica.array<10x20x!modelica.real>
        %2 = modelica.der %1 : !modelica.real -> !modelica.real
        %3 = modelica.constant #modelica.real<3.0>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        modelica.equation_sides %4, %5 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>} : !modelica.equation
}
