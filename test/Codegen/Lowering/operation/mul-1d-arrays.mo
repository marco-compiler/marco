// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.int

function Integers
    input Integer[3] x;
    input Integer[3] y;
    output Integer z;
algorithm
    z := x * y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.real>) -> !modelica.real

function Reals
    input Real[3] x;
    input Real[3] y;
    output Real z;
algorithm
    z := x * y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.real>) -> !modelica.real

function IntegerReal
    input Integer[3] x;
    input Real[3] y;
    output Real z;
algorithm
    z := x * y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.int>) -> !modelica.real

function RealInteger
    input Real[3] x;
    input Integer[3] y;
    output Real z;
algorithm
    z := x * y;
end RealInteger;
