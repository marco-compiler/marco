// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK: bmodelica.neg %{{.*}} : !bmodelica.int -> !bmodelica.int

function Integers
    input Integer x;
    output Integer y;
algorithm
    y := -x;
end Integers;

// CHECK-LABEL: @Real
// CHECK: bmodelica.neg %{{.*}} : !bmodelica.real -> !bmodelica.real

function Reals
    input Real x;
    output Real y;
algorithm
    y := -x;
end Reals;
