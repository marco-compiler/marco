// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @sizeArray
// CHECK: modelica.size
// CHECK-SAME: !modelica.array<?x?x!modelica.real> -> !modelica.array<2x!modelica.int>

function sizeArray
    input Real[:,:] x;
    output Integer[2] y;

algorithm
    y := size(x);
end sizeArray;

// CHECK-LABEL: @sizeDimension
// CHECK: modelica.size
// CHECK-SAME: (!modelica.array<?x?x!modelica.real>, index) -> !modelica.int

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;

algorithm
    y := size(x, n);
end sizeDimension;
