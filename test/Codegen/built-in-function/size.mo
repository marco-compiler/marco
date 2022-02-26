// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @sizeArray
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.real>
// CHECK: modelica.size %arg0 : !modelica.array<?x?x!modelica.real> -> !modelica.array<heap, 2x!modelica.int>

function sizeArray
    input Real[:,:] x;
    output Integer[2] y;

algorithm
    y := size(x);
end sizeArray;

// CHECK-LABEL: @sizeDimension
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.real>
// CHECK-SAME: %arg1 : !modelica.int
// CHECK: %[[C1:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<1> : !modelica.int
// CHECK: %[[INDEX:[a-zA-Z0-9]*]] = modelica.sub %arg1, %[[C1]] : (!modelica.int, !modelica.int) -> index
// CHECK: modelica.size %arg0[%[[INDEX]]] : (!modelica.array<?x?x!modelica.real>, index) -> !modelica.int

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;

algorithm
    y := size(x, n);
end sizeDimension;
