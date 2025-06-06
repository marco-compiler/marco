// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// CHECK-LABEL: @scalarVariable

bmodelica.model @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = []
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[der_x]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK-NOT: bmodelica.equation_template

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:  bmodelica.equation_instance %[[t0]]

    // CHECK-NOT: bmodelica.equation_instance
}

// -----

// CHECK-LABEL: @arrayVariable

bmodelica.model @arrayVariable {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
    // CHECK-DAG:       %[[der_x_view:.*]] = bmodelica.tensor_view %[[der_x]][%[[index]]]
    // CHECK-DAG:       %[[der_x_extract:.*]] = bmodelica.tensor_extract %[[der_x_view]][]
    // CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[der_x_extract]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}

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

    // CHECK:       %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "eq1"}
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[index:.*]] = bmodelica.constant 1 : index
    // CHECK-DAG:       %[[der_x_view:.*]] = bmodelica.tensor_view %[[der_x]][%[[index]]]
    // CHECK-DAG:       %[[der_x_extract:.*]] = bmodelica.tensor_extract %[[der_x_view]][]
    // CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[der_x_extract]]
    // CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}

    // CHECK-NOT: bmodelica.equation_template

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
    }

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:  bmodelica.equation_instance %[[t0]]
    // CHECK-NEXT:  bmodelica.equation_instance %[[t1]]

    // CHECK-NOT: bmodelica.equation_instance
}