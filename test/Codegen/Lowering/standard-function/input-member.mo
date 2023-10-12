// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @inputBooleanScalar
// CHECK: modelica.variable @x : !modelica.variable<!modelica.bool, input>

function inputBooleanScalar
    input Boolean x;
algorithm
end inputBooleanScalar;


// CHECK-LABEL: @inputIntegerScalar
// CHECK: modelica.variable @x : !modelica.variable<!modelica.int, input>

function inputIntegerScalar
    input Integer x;
algorithm
end inputIntegerScalar;


// CHECK-LABEL: @inputRealScalar
// CHECK: modelica.variable @x : !modelica.variable<!modelica.real, input>

function inputRealScalar
    input Real x;
algorithm
end inputRealScalar;


// CHECK-LABEL: @inputBooleanStaticArray
// CHECK: modelica.variable @x : !modelica.variable<3x2x!modelica.bool, input>

function inputBooleanStaticArray
    input Boolean[3,2] x;
algorithm
end inputBooleanStaticArray;


// CHECK-LABEL: @inputBooleanDynamicArray
// CHECK: modelica.variable @x : !modelica.variable<?x?x!modelica.bool, input>

function inputBooleanDynamicArray
    input Boolean[:,:] x;
algorithm
end inputBooleanDynamicArray;


// CHECK-LABEL: @inputIntegerStaticArray
// CHECK: modelica.variable @x
// CHECK-SAME: !modelica.variable<3x2x!modelica.int, input>

function inputIntegerStaticArray
    input Integer[3,2] x;
algorithm
end inputIntegerStaticArray;


// CHECK-LABEL: @inputIntegerDynamicArray
// CHECK: modelica.variable @x
// CHECK-SAME: !modelica.variable<?x?x!modelica.int, input>

function inputIntegerDynamicArray
    input Integer[:,:] x;
algorithm
end inputIntegerDynamicArray;


// CHECK-LABEL: @inputRealStaticArray
// CHECK: modelica.variable @x : !modelica.variable<3x2x!modelica.real, input>

function inputRealStaticArray
    input Real[3,2] x;
algorithm
end inputRealStaticArray;


// CHECK-LABEL: @inputRealDynamicArray
// CHECK: modelica.variable @x : !modelica.variable<?x?x!modelica.real, input>

function inputRealDynamicArray
    input Real[:,:] x;
algorithm
end inputRealDynamicArray;
