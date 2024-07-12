// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: bmodelica.assert
// CHECK-SAME: level = 2 : i64
// CHECK-SAME: message = "y was not set to 2"


function foo
  input Integer n1;
  input Integer n2;
  output Real y;
algorithm
  y := n1 + n2;
  assert(y == 5, "y was not set to 2");
end foo;

model M
  Real x;
equation
  x = foo(2, 3);
end M;
