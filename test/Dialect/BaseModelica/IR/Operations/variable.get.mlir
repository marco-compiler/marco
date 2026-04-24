// RUN: modelica-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @Scalar

bmodelica.function @Scalar {
    bmodelica.variable @var : !bmodelica.variable<!bmodelica.real>

    bmodelica.algorithm {
        %0 = bmodelica.variable.get @var : !bmodelica.real

        // CHECK: %{{.*}} = bmodelica.variable.get @var : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @Array

bmodelica.function @Array {
    bmodelica.variable @var : !bmodelica.variable<2x3x!bmodelica.real>

    bmodelica.algorithm {
        %0 = bmodelica.variable.get @var : tensor<2x3x!bmodelica.real>
        
        // CHECK: %{{.*}} = bmodelica.variable.get @var : tensor<2x3x!bmodelica.real>
    }
}
