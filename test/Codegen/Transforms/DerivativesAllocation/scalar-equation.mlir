// RUN: modelica-opt %s --split-input-file --allocate-derivatives | FileCheck %s

// Scalar variable.

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x>]
// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.real>
// CHECK-DAG: modelica.variable @der_x : !modelica.variable<!modelica.real>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>

    %t0 = modelica.equation_template inductions = [] {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.der %0 : !modelica.real -> !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-NEXT:      %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = modelica.equation_side %[[der_x]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.der %0 : !modelica.real -> !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      modelica.yield %[[zero]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>

    %t0 = modelica.equation_template inductions = [] {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.der %0 : !modelica.real -> !modelica.real
        %2 = modelica.equation_side %0 : tuple<!modelica.real>
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        modelica.equation_sides %2, %3 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
    }
}

// -----

// Array variable.
// All indices are derived.

// Check variable declaration and derivatives map.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,1]}>
// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x, #[[index_set]]>]
// CHECK-DAG: modelica.variable @x : !modelica.variable<2x!modelica.real>
// CHECK-DAG: modelica.variable @der_x : !modelica.variable<2x!modelica.real>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    %t1 = modelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 1 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}

// -----

// Check variable usage.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[0,1]}>
// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x, #[[index_set]]>]
// CHECK:       modelica.equation_template inductions = [] attributes {id = "eq0"} {
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-DAG:       %[[index:.*]] = modelica.constant 0 : index
// CHECK-DAG:       %[[der_x_subscription:.*]] = modelica.subscription %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[der_x_load:.*]] = modelica.load %[[der_x_subscription]][]
// CHECK:           %[[lhs:.*]] = modelica.equation_side %[[der_x_load]]
// CHECK:           modelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }
// CHECK:       modelica.equation_template inductions = [] attributes {id = "eq1"} {
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-DAG:       %[[index:.*]] = modelica.constant 1 : index
// CHECK-DAG:       %[[der_x_subscription:.*]] = modelica.subscription %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[der_x_load:.*]] = modelica.load %[[der_x_subscription]][]
// CHECK:           %[[lhs:.*]] = modelica.equation_side %[[der_x_load]]
// CHECK:           modelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    %t1 = modelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 1 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       modelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      %[[array:.*]] = modelica.array_broadcast %[[zero]] : !modelica.real -> !modelica.array<2x!modelica.real>
// CHECK-NEXT:      modelica.yield %[[array]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<2x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    %t1 = modelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = modelica.variable_get @x : !modelica.array<2x!modelica.real>
        %1 = modelica.constant 1 : index
        %2 = modelica.load %0[%1] : !modelica.array<2x!modelica.real>
        %3 = modelica.der %2 : !modelica.real -> !modelica.real
        %4 = modelica.constant #modelica.real<0.0>
        %5 = modelica.equation_side %3 : tuple<!modelica.real>
        %6 = modelica.equation_side %4 : tuple<!modelica.real>
        modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
        modelica.equation_instance %t1 : !modelica.equation
    }
}

// -----

// Array variable.
// Not all indices are derived.

// Check variable declaration and derivatives map.

// CHECK: #[[index_set:.*]] = #modeling<index_set {[3,4]}>
// CHECK-LABEL: @Test
// CHECK-SAME: derivatives_map = [#modelica<var_derivative @x, @der_x, #[[index_set]]>]
// CHECK-DAG: modelica.variable @x : !modelica.variable<10x!modelica.real>
// CHECK-DAG: modelica.variable @der_x : !modelica.variable<10x!modelica.real>

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.constant 4 : index
        %3 = modelica.load %0[%1] : !modelica.array<10x!modelica.real>
        %4 = modelica.load %0[%2] : !modelica.array<10x!modelica.real>
        %5 = modelica.der %3 : !modelica.real -> !modelica.real
        %6 = modelica.der %4 : !modelica.real -> !modelica.real
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
    }
}

// -----

// Check equations for non-derived indices.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[zero:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK-DAG:       %[[load:.*]] = modelica.load %[[der_x]][%[[i0]]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[zero]]
// CHECK-NEXT:      modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-DAG:  modelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [5,9]>}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<10x!modelica.real>

    %t0 = modelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = modelica.variable_get @x : !modelica.array<10x!modelica.real>
        %1 = modelica.constant 3 : index
        %2 = modelica.constant 4 : index
        %3 = modelica.load %0[%1] : !modelica.array<10x!modelica.real>
        %4 = modelica.load %0[%2] : !modelica.array<10x!modelica.real>
        %5 = modelica.der %3 : !modelica.real -> !modelica.real
        %6 = modelica.der %4 : !modelica.real -> !modelica.real
        %7 = modelica.equation_side %5 : tuple<!modelica.real>
        %8 = modelica.equation_side %6 : tuple<!modelica.real>
        modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
    }

    modelica.main_model {
        modelica.equation_instance %t0 : !modelica.equation
    }
}
