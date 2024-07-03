// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       bmodelica.function @Foo {
// CHECK:           bmodelica.algorithm {
// CHECK-DAG:           %[[one:.*]] = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:           %[[minus_one:.*]] = bmodelica.constant -1 : index
// CHECK-DAG:           %[[subscript:.*]] = bmodelica.add %[[one]], %[[minus_one]]
// CHECK-DAG:           %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.real
// CHECK-NEXT:          bmodelica.variable_component_set @r[%[[subscript]]]::@x, %[[x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input Real x;
    output R[3] r;
algorithm
    r[1].x := x;
end Foo;
