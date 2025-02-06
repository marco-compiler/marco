// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// Check variable declaration and derivatives map.

// CHECK-LABEL: @Test
// CHECK-SAME: der = [<@x, @der_x, {[3,5][12,14]}>]
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>
// CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<10x20x!bmodelica.real>

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>}
    }
}

// -----

// Check variable usage.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] attributes {id = "t0"} {
// CHECK:           %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:           %[[view:.*]] = bmodelica.tensor_view %[[der_x]][%[[i0]], %[[i1]]]
// CHECK:           %[[extract:.*]] = bmodelica.tensor_extract %[[view]][]
// CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }
// CHECK:       bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][12,14]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>}
    }
}

// -----

// Check start value.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @der_x {
// CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[zero]] : !bmodelica.real -> tensor<10x20x!bmodelica.real>
// CHECK-NEXT:      bmodelica.yield %[[tensor]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>}
    }
}

// -----

// Check equations for non-derived indices.

// CHECK-LABEL: @Test
// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]] {
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[i0]], %[[i1]]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2][0,19]>}
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][0,11]>}
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [3,5][15,19]>}
// CHECK-DAG:  bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [6,9][0,19]>}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0 {indices = #modeling<multidim_range [3,5][12,14]>}
    }
}
