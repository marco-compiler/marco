// RUN: modelica-opt %s --split-input-file --derivatives-initialization | FileCheck %s

// CHECK-LABEL: @Partial1DArrayVariable

bmodelica.model @Partial1DArrayVariable der = [<@x, @der_x, {[1,2], [5,7]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]]
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[i0]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:      bmodelica.equation_instance %[[t0]], indices = {[0,0],[3,4],[8,9]}
}

// -----

// CHECK-LABEL: @Partial2DArrayVariable

bmodelica.model @Partial2DArrayVariable der = [<@x, @der_x, {[3,5][12,14]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x20x!bmodelica.real>

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]], %[[i1:.*]]]
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[der_x]][%[[i0]], %[[i1]]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %[[rhs]]

    // CHECK:       bmodelica.dynamic
    // CHECK-NEXT:      bmodelica.equation_instance %[[t0]], indices = {[0,2][0,19],[3,5][0,11],[3,5][15,19],[6,9][0,19]}
}