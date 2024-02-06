// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       modelica.function @Foo {
// CHECK:           modelica.algorithm {
// CHECK-DAG:           %[[r:.*]] = modelica.variable_get @r : !modelica.array<3x!modelica<record @R>>
// CHECK-DAG:           %[[x:.*]] = modelica.variable_get @x : !modelica.array<3x!modelica.real>
// CHECK-NEXT:          modelica.component_set %[[r]], @x, %[[x]] : !modelica.array<3x!modelica<record @R>>, !modelica.array<3x!modelica.real>
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input Real[3] x;
    output R[3] r;
algorithm
    r.x := x;
end Foo;
