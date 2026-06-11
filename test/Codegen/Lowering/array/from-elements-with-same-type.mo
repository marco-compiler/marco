// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %[[v1:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG: %[[v2:.*]] = bmodelica.constant #bmodelica<int 2>
// CHECK-DAG: %[[v3:.*]] = bmodelica.constant #bmodelica<int 3>
// CHECK-DAG: %[[v4:.*]] = bmodelica.constant #bmodelica<int 4>
// CHECK-DAG: %[[v5:.*]] = bmodelica.constant #bmodelica<int 5>
// CHECK: bmodelica.tensor.from_elements %[[v1]], %[[v2]], %[[v3]], %[[v4]], %[[v5]]
// CHECK-SAME: -> tensor<5x!bmodelica.int>

function foo
  output Integer[5] x;
algorithm
  x := {1, 2, 3, 4, 5};
end foo;
