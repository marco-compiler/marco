// RUN: marco %s --omc-bypass --emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.sub %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.int>, !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.int>

function Integers
    input Integer[3,5] x;
    input Integer[3,5] y;
    output Integer[3,5] z;
algorithm
    z := x - y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.sub %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.real>, !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>

function Reals
    input Real[3,5] x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x - y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.sub %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.int>, !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>

function IntegerReal
    input Integer[3,5] x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x - y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.sub %{{.*}}, %{{.*}} : (!modelica.array<3x5x!modelica.real>, !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.real>

function RealInteger
    input Real[3,5] x;
    input Integer[3,5] y;
    output Real[3,5] z;
algorithm
    z := x - y;
end RealInteger;
