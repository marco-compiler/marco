// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.int, tensor<3x5x!bmodelica.int>) -> tensor<3x5x!bmodelica.int>

function Integers
    input Integer x;
    input Integer[3,5] y;
    output Integer[3,5] z;
algorithm
    z := x .^ y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.real, tensor<3x5x!bmodelica.real>) -> tensor<3x5x!bmodelica.real>

function Reals
    input Real x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x .^ y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.int, tensor<3x5x!bmodelica.real>) -> tensor<3x5x!bmodelica.real>

function IntegerReal
    input Integer x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x .^ y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: bmodelica.pow_ew %{{.*}}, %{{.*}} : (!bmodelica.real, tensor<3x5x!bmodelica.int>) -> tensor<3x5x!bmodelica.real>

function RealInteger
    input Real x;
    input Integer[3,5] y;
    output Real[3,5] z;
algorithm
    z := x .^ y;
end RealInteger;
