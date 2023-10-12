// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK: modelica.neg %{{.*}} : !modelica.int -> !modelica.int

function Integers
    input Integer x;
    output Integer y;
algorithm
    y := -x;
end Integers;

// CHECK-LABEL: @Real
// CHECK: modelica.neg %{{.*}} : !modelica.real -> !modelica.real

function Reals
    input Real x;
    output Real y;
algorithm
    y := -x;
end Reals;
