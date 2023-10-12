// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       modelica.function @Foo {
// CHECK:           modelica.algorithm {
// CHECK-NEXT:          %[[r:.*]] = modelica.variable_get @r : !modelica.record<@R>
// CHECK-NEXT:          %[[r_x:.*]] = modelica.component_get %[[r]], @x : !modelica.record<@R> -> !modelica.real
// CHECK-NEXT:          modelica.variable_set @x, %[[r_x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input R r;
    output Real x;
algorithm
    x := r.x;
end Foo;
