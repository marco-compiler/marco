// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.int, !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.int>

function Integers
    input Integer x;
    input Integer[3,5] y;
    output Integer[3,5] z;
algorithm
    z := x * y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.real, !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>

function Reals
    input Real x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x * y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.int, !modelica.array<3x5x!modelica.real>) -> !modelica.array<3x5x!modelica.real>

function IntegerReal
    input Integer x;
    input Real[3,5] y;
    output Real[3,5] z;
algorithm
    z := x * y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.real, !modelica.array<3x5x!modelica.int>) -> !modelica.array<3x5x!modelica.real>

function RealInteger
    input Real x;
    input Integer[3,5] y;
    output Real[3,5] z;
algorithm
    z := x * y;
end RealInteger;
