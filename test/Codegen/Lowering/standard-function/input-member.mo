// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @inputBooleanScalar
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.bool, input>

function inputBooleanScalar
    input Boolean x;
algorithm
end inputBooleanScalar;


// CHECK-LABEL: @inputIntegerScalar
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>

function inputIntegerScalar
    input Integer x;
algorithm
end inputIntegerScalar;


// CHECK-LABEL: @inputRealScalar
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>

function inputRealScalar
    input Real x;
algorithm
end inputRealScalar;


// CHECK-LABEL: @inputBooleanStaticArray
// CHECK: bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.bool, input>

function inputBooleanStaticArray
    input Boolean[3,2] x;
algorithm
end inputBooleanStaticArray;


// CHECK-LABEL: @inputBooleanDynamicArray
// CHECK: bmodelica.variable @x : !bmodelica.variable<?x?x!bmodelica.bool, input>

function inputBooleanDynamicArray
    input Boolean[:,:] x;
algorithm
end inputBooleanDynamicArray;


// CHECK-LABEL: @inputIntegerStaticArray
// CHECK: bmodelica.variable @x
// CHECK-SAME: !bmodelica.variable<3x2x!bmodelica.int, input>

function inputIntegerStaticArray
    input Integer[3,2] x;
algorithm
end inputIntegerStaticArray;


// CHECK-LABEL: @inputIntegerDynamicArray
// CHECK: bmodelica.variable @x
// CHECK-SAME: !bmodelica.variable<?x?x!bmodelica.int, input>

function inputIntegerDynamicArray
    input Integer[:,:] x;
algorithm
end inputIntegerDynamicArray;


// CHECK-LABEL: @inputRealStaticArray
// CHECK: bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.real, input>

function inputRealStaticArray
    input Real[3,2] x;
algorithm
end inputRealStaticArray;


// CHECK-LABEL: @inputRealDynamicArray
// CHECK: bmodelica.variable @x : !bmodelica.variable<?x?x!bmodelica.real, input>

function inputRealDynamicArray
    input Real[:,:] x;
algorithm
end inputRealDynamicArray;
