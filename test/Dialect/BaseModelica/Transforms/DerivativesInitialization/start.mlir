// RUN: modelica-opt %s --split-input-file --derivatives-initialization | FileCheck %s

// CHECK-LABEL: @scalarVariable

bmodelica.model @scalarVariable der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    // CHECK:         bmodelica.start @der_x
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT:      bmodelica.yield %[[zero]]
}

// -----

// CHECK-LABEL: @ArrayVariable1d

bmodelica.model @ArrayVariable1d der = [<@x, @der_x, {[0,1]}>] {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>

    // CHECK:         bmodelica.start @der_x
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[zero]] : !bmodelica.real -> tensor<2x!bmodelica.real>
    // CHECK-NEXT:      bmodelica.yield %[[tensor]]
}

// -----

// CHECK-LABEL: @ArrayVariable2d

bmodelica.model @ArrayVariable2d der = [<@x, @der_x, {[3,5][12,14]}>] {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<10x20x!bmodelica.real>

    // CHECK:       bmodelica.start @der_x
    // CHECK-NEXT:      %[[zero:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
    // CHECK-NEXT:      %[[tensor:.*]] = bmodelica.tensor_broadcast %[[zero]] : !bmodelica.real -> tensor<10x20x!bmodelica.real>
    // CHECK-NEXT:      bmodelica.yield %[[tensor]]
}