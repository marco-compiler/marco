// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @minArray
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.real>
// CHECK: modelica.min %arg0 : !modelica.array<?x?x!modelica.real> -> !modelica.real

function minArray
    input Real[:,:] x;
    output Real y;

algorithm
    y := min(x);
end minArray;

// CHECK-LABEL: @minScalars
// CHECK-SAME: %arg0 : !modelica.real
// CHECK-SAME: %arg1 : !modelica.real
// CHECK: modelica.min %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real

function minScalars
    input Real x;
    input Real y;
    output Real z;

algorithm
    z := min(x, y);
end minScalars;
