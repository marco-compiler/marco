// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @minArray
// CHECK: modelica.max
// CHECK-SAME: !modelica.array<?x?x!modelica.real> -> !modelica.real

function minArray
    input Real[:,:] x;
    output Real y;
algorithm
    y := max(x);
end minArray;

// CHECK-LABEL: @minScalars
// CHECK: modelica.max
// CHECK-SAME: (!modelica.real, !modelica.real) -> !modelica.real

function minScalars
    input Real x;
    input Real y;
    output Real z;
algorithm
    z := max(x, y);
end minScalars;
