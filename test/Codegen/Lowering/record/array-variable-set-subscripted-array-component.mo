// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK-DAG:           %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:           %[[minus_one:.*]] = bmodelica.constant -1 : index
// CHECK-DAG:           %[[subscript:.*]] = bmodelica.add %[[one]], %[[minus_one]]
// CHECK-DAG:           %[[x:.*]] = bmodelica.variable_get @x : tensor<3x!bmodelica.real>
// CHECK:               bmodelica.variable_component_set @r::@x[%[[subscript]]], %[[x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real[2] x;
end R;

function Foo
    input Real[3] x;
    output R[3] r;
algorithm
    r.x[1] := x;
end Foo;
