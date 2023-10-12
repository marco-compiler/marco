// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK:       modelica.function @Foo {
// CHECK:           modelica.algorithm {
// CHECK-NEXT:          %[[r:.*]] = modelica.variable_get @r : !modelica.array<3x!modelica.record<@R>>
// CHECK-NEXT:          %[[r_x:.*]] = modelica.component_get %[[r]], @x : !modelica.array<3x!modelica.record<@R>> -> !modelica.array<3x!modelica.real>
// CHECK-NEXT:          modelica.variable_set @x, %[[r_x]]
// CHECK-NEXT:      }
// CHECK-NEXT:  }

record R
    Real x;
end R;

function Foo
    input R[3] r;
    output Real[3] x;
algorithm
    x := r.x;
end Foo;
