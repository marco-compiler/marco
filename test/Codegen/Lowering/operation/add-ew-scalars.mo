// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.int

function Integers
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := x .+ y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (!modelica.real, !modelica.real) -> !modelica.real

function Reals
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := x .+ y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.real

function IntegerReal
    input Integer x;
    input Real y;
    output Real z;
algorithm
    z := x .+ y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (!modelica.real, !modelica.int) -> !modelica.real

function RealInteger
    input Real x;
    input Integer y;
    output Real z;
algorithm
    z := x .+ y;
end RealInteger;
