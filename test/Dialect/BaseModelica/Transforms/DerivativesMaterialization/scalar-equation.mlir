// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// Scalar variable.

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation_template inductions = [] attributes {id = "t0"} {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[der_x]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      bmodelica.yield %[[zero]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
    }
}

// -----

// Array variable.
// All indices are derived.

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x, {[0,1]}>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x, {[0,1]}>]
// CHECK:       bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[der_x_view:.*]] = bmodelica.tensor_view %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[der_x_extract:.*]] = bmodelica.tensor_extract %[[der_x_view]][]
// CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[der_x_extract]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }
// CHECK:       bmodelica.equation_template inductions = [] attributes {id = "eq1"} {
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 1 : index
// CHECK-DAG:       %[[der_x_view:.*]] = bmodelica.tensor_view %[[der_x]][%[[index]]]
// CHECK-DAG:       %[[der_x_extract:.*]] = bmodelica.tensor_extract %[[der_x_view]][]
// CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[der_x_extract]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[zero]] : !bmodelica.real -> tensor<2x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[tensor]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
    }
}

// -----

// Array variable.
// Not all indices are derived.

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x, {[3,4]}>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<10x!bmodelica.real>

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.constant 4 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.real>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<10x!bmodelica.real>
        %5 = bmodelica.der %3 : !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.der %4 : !bmodelica.real -> !bmodelica.real
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
    }
}

// -----

// Check equations for non-derived indices.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[i0]]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [5,9]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "eq0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.constant 4 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.real>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<10x!bmodelica.real>
        %5 = bmodelica.der %3 : !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.der %4 : !bmodelica.real -> !bmodelica.real
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 : !bmodelica.equation
    }
}
