// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK:       modelica.function @Foo {
// CHECK:           modelica.algorithm {
// CHECK-DAG:           %[[r:.*]] = modelica.variable_get @r : !modelica.record<@R>
// CHECK-DAG:           %[[x:.*]] = modelica.variable_get @x : !modelica.real
// CHECK-NEXT:          modelica.component_set %[[r]], @x, %[[x]] : !modelica.record<@R>, !modelica.real
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input Real x;
    output R r;
algorithm
    r.x := x;
end Foo;
