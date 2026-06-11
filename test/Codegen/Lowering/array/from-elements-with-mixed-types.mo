// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %[[v1:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-DAG: %[[v2:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-DAG: %[[v3:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
// CHECK-DAG: %[[v4:.*]] = bmodelica.constant #bmodelica<int 4>
// CHECK-DAG: %[[v5:.*]] = bmodelica.constant #bmodelica<real 5.000000e+00>
// CHECK: bmodelica.tensor.from_elements %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]]
// CHECK-SAME: -> tensor<5x!bmodelica.real>

function foo
  output Integer[5] x;
algorithm
  x := {1.0, 2, 3.0, 4, 5.0};
end foo;
