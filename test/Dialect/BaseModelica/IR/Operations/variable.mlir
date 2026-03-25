// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Scalar
// CHECK: bmodelica.variable @var : !bmodelica.variable<!bmodelica.real>

bmodelica.model @Scalar {
    bmodelica.variable @var : !bmodelica.variable<!bmodelica.real>
}

// -----

// CHECK-LABEL: @Array
// CHECK: bmodelica.variable @var : !bmodelica.variable<2x3x!bmodelica.real>

bmodelica.model @Array {
    bmodelica.variable @var : !bmodelica.variable<2x3x!bmodelica.real>
}

// -----

// CHECK-LABEL: @FreeDimension
// CHECK: bmodelica.variable @var : !bmodelica.variable<?x!bmodelica.real> [free]

bmodelica.model @FreeDimension {
    bmodelica.variable @var : !bmodelica.variable<?x!bmodelica.real> [free]
}

// -----

// CHECK-LABEL: @FixedDimension
// CHECK:       bmodelica.variable @var : !bmodelica.variable<?x!bmodelica.real> [fixed] {
// CHECK-NEXT:      %[[cst:.*]] = arith.constant 1 : index
// CHECK-NEXT:      bmodelica.yield %[[cst]] : index
// CHECK-NEXT: }

bmodelica.model @FixedDimension {
    bmodelica.variable @var : !bmodelica.variable<?x!bmodelica.real> [fixed] {
        %0 = arith.constant 1 : index
        bmodelica.yield %0 : index
    }
}
