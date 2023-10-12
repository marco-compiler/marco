// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (!modelica.array<3x3x!modelica.int>, !modelica.int) -> !modelica.array<3x3x!modelica.int>

function Integers
    input Integer[3,3] x;
    input Integer y;
    output Integer[3,3] z;
algorithm
    z := x ^ y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (!modelica.array<3x3x!modelica.real>, !modelica.real) -> !modelica.array<3x3x!modelica.real>

function Reals
    input Real[3,3] x;
    input Real y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (!modelica.array<3x3x!modelica.int>, !modelica.real) -> !modelica.array<3x3x!modelica.int>

function IntegerReal
    input Integer[3,3] x;
    input Real y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (!modelica.array<3x3x!modelica.real>, !modelica.int) -> !modelica.array<3x3x!modelica.real>

function RealInteger
    input Real[3,3] x;
    input Integer y;
    output Real[3,3] z;
algorithm
    z := x ^ y;
end RealInteger;
