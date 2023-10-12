// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.int>, !modelica.array<5x4x!modelica.int>) -> !modelica.array<3x4x!modelica.int>

function Integers
    input Integer[3,5] x;
    input Integer[5,4] y;
    output Integer[3,4] z;
algorithm
    z := x * y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.real>, !modelica.array<5x4x!modelica.real>) -> !modelica.array<3x4x!modelica.real>

function Reals
    input Real[3,5] x;
    input Real[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.int>, !modelica.array<5x4x!modelica.real>) -> !modelica.array<3x4x!modelica.real>

function IntegerReal
    input Integer[3,5] x;
    input Real[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.real>, !modelica.array<5x4x!modelica.int>) -> !modelica.array<3x4x!modelica.real>

function RealInteger
    input Real[3,5] x;
    input Integer[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end RealInteger;
