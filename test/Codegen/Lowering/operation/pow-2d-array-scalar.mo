// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: bmodelica.pow %{{.*}}, %{{.*}} : (!bmodelica.array<3x3x!bmodelica.int>, !bmodelica.int) -> !bmodelica.array<3x3x!bmodelica.int>

function Integers
    input Integer[3,3] x;
    input Integer y;
    output Integer[3,3] z;
algorithm
    z := x ^ y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: bmodelica.pow %{{.*}}, %{{.*}} : (!bmodelica.array<3x3x!bmodelica.real>, !bmodelica.real) -> !bmodelica.array<3x3x!bmodelica.real>

function Reals
    input Real[3,3] x;
    input Real y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: bmodelica.pow %{{.*}}, %{{.*}} : (!bmodelica.array<3x3x!bmodelica.int>, !bmodelica.real) -> !bmodelica.array<3x3x!bmodelica.int>

function IntegerReal
    input Integer[3,3] x;
    input Real y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: bmodelica.pow %{{.*}}, %{{.*}} : (!bmodelica.array<3x3x!bmodelica.real>, !bmodelica.int) -> !bmodelica.array<3x3x!bmodelica.real>

function RealInteger
    input Real[3,3] x;
    input Integer y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end RealInteger;
