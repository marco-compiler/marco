// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK: bmodelica.neg %{{.*}} : !bmodelica.array<3x5x!bmodelica.int> -> !bmodelica.array<3x5x!bmodelica.int>

function Integers
    input Integer[3,5] x;
    output Integer[3,5] y;
algorithm
    y := -x;
end Integers;

// CHECK-LABEL: @Real
// CHECK: bmodelica.neg %{{.*}} : !bmodelica.array<3x5x!bmodelica.real> -> !bmodelica.array<3x5x!bmodelica.real>

function Reals
    input Real[3,5] x;
    output Real[3,5] y;
algorithm
    y := -x;
end Reals;
