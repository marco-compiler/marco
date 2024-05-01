// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int

function Integers
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := x .^ y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real

function Reals
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := x .^ y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real

function IntegerReal
    input Integer x;
    input Real y;
    output Real z;
algorithm
    z := x .^ y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real

function RealInteger
    input Real x;
    input Integer y;
    output Real z;
algorithm
    z := x .^ y;
end RealInteger;
