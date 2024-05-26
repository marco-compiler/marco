// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x!bmodelica.int>, tensor<3x!bmodelica.int>) -> !bmodelica.int

function Integers
    input Integer[3] x;
    input Integer[3] y;
    output Integer z;
algorithm
    z := x * y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x!bmodelica.real>, tensor<3x!bmodelica.real>) -> !bmodelica.real

function Reals
    input Real[3] x;
    input Real[3] y;
    output Real z;
algorithm
    z := x * y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x!bmodelica.int>, tensor<3x!bmodelica.real>) -> !bmodelica.real

function IntegerReal
    input Integer[3] x;
    input Real[3] y;
    output Real z;
algorithm
    z := x * y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x!bmodelica.real>, tensor<3x!bmodelica.int>) -> !bmodelica.real

function RealInteger
    input Real[3] x;
    input Integer[3] y;
    output Real z;
algorithm
    z := x * y;
end RealInteger;
