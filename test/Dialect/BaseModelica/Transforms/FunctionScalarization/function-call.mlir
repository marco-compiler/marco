// RUN: modelica-opt %s --scalarize | FileCheck %s

// CHECK: #[[map:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @caller
// CHECK-SAME:  (%[[arg0:.*]]: tensor<3x!bmodelica.real>)
// CHECK:       %[[result:.*]] = bmodelica.alloc
// CHECK:       %[[lb:.*]] = arith.constant 0 : index
// CHECK:       %[[dim:.*]] = arith.constant 0 : index
// CHECK:       %[[ub:.*]] = tensor.dim %[[arg0]], %[[dim]]
// CHECK:       affine.for %[[i:.*]] = #[[map]](%[[lb]]) to #[[map]](%[[ub]]) {
// CHECK:           %[[extract:.*]] = bmodelica.tensor_extract %[[arg0]][%[[i]]]
// CHECK:           %[[scalarResult:.*]] = bmodelica.call @callee(%[[extract]]) : (!bmodelica.real) -> !bmodelica.real
// CHECK:           bmodelica.store %[[result]][%[[i]]], %[[scalarResult]]
// CHECK-NEXT:  }
// CHECK:       %[[result_tensor:.*]] = bmodelica.array_to_tensor %[[result]]
// CHECK:       return %[[result_tensor]]

bmodelica.function @callee {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

func.func @caller(%arg0: tensor<3x!bmodelica.real>) -> (tensor<3x!bmodelica.real>) {
    %result = bmodelica.call @callee(%arg0) : (tensor<3x!bmodelica.real>) -> (tensor<3x!bmodelica.real>)
    return %result : tensor<3x!bmodelica.real>
}
