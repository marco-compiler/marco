// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integer
// CHECK: modelica.neg %{{.*}} : !modelica.array<3x5x!modelica.int> -> !modelica.array<3x5x!modelica.int>

function Integers
    input Integer[3,5] x;
    output Integer[3,5] y;
algorithm
    y := -x;
end Integers;

// CHECK-LABEL: @Real
// CHECK: modelica.neg %{{.*}} : !modelica.array<3x5x!modelica.real> -> !modelica.array<3x5x!modelica.real>

function Reals
    input Real[3,5] x;
    output Real[3,5] y;
algorithm
    y := -x;
end Reals;
