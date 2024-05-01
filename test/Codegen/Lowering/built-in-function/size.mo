// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @sizeArray
// CHECK: bmodelica.size
// CHECK-SAME: !bmodelica.array<?x?x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.int>

function sizeArray
    input Real[:,:] x;
    output Integer[2] y;
algorithm
    y := size(x);
end sizeArray;

// CHECK-LABEL: @sizeDimension
// CHECK: bmodelica.size
// CHECK-SAME: (!bmodelica.array<?x?x!bmodelica.real>, index) -> !bmodelica.int

function sizeDimension
    input Real[:,:] x;
    input Integer n;
    output Integer[2] y;
algorithm
    y := size(x, n);
end sizeDimension;
