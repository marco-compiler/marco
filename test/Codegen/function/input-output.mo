// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @inputBooleanScalar
// CHECK-SAME: %arg0 : !modelica.bool

function inputBooleanScalar
    input Boolean x;

algorithm
end inputBooleanScalar;


// CHECK-LABEL: @inputIntegerScalar
// CHECK-SAME: %arg0 : !modelica.int

function inputIntegerScalar
    input Integer x;

algorithm
end inputIntegerScalar;


// CHECK-LABEL: @inputRealScalar
// CHECK-SAME: %arg0 : !modelica.real

function inputRealScalar
    input Real x;

algorithm
end inputRealScalar;


// CHECK-LABEL: @inputBooleanStaticArray
// CHECK-SAME: %arg0 : !modelica.array<3x2x!modelica.bool>

function inputBooleanStaticArray
    input Boolean[3,2] x;

algorithm
end inputBooleanStaticArray;


// CHECK-LABEL: @inputBooleanDynamicArray
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.bool>

function inputBooleanDynamicArray
    input Boolean[:,:] x;

algorithm
end inputBooleanDynamicArray;


// CHECK-LABEL: @inputIntegerStaticArray
// CHECK-SAME: %arg0 : !modelica.array<3x2x!modelica.int>

function inputIntegerStaticArray
    input Integer[3,2] x;

algorithm
end inputIntegerStaticArray;


// CHECK-LABEL: @inputIntegerDynamicArray
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.int>

function inputIntegerDynamicArray
    input Integer[:,:] x;

algorithm
end inputIntegerDynamicArray;


// CHECK-LABEL: @inputRealStaticArray
// CHECK-SAME: %arg0 : !modelica.array<3x2x!modelica.real>

function inputRealStaticArray
    input Real[3,2] x;

algorithm
end inputRealStaticArray;


// CHECK-LABEL: @inputRealDynamicArray
// CHECK-SAME: %arg0 : !modelica.array<?x?x!modelica.real>

function inputRealDynamicArray
    input Real[:,:] x;

algorithm
end inputRealDynamicArray;


// CHECK-LABEL: @arrayCopy
// CHECK-SAME: %arg0 : !modelica.array<?x!modelica.int>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<heap, ?x!modelica.int>
// CHECK: modelica.member_store %[[Y]], %arg0

function arrayCopy
    input Integer[:] x;
    output Integer[:] y;

algorithm
    y := x;
end arrayCopy;


// CHECK-LABEL: @constantOutput
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<stack, !modelica.int>
// CHECK: %[[CONST:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<10> : !modelica.int
// CHECK: modelica.member_store %[[Y]], %[[CONST]]

function constantOutput
    output Integer y;

algorithm
    y := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK-SAME: %arg0 : !modelica.int
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<stack, !modelica.real>
// CHECK: modelica.member_store %[[Y]], %arg0

function castIntegerToReal
    input Integer x;
    output Real y;

algorithm
    y := x;
end castIntegerToReal;


// CHECK-LABEL: @sizeDependingOnIntegerInput
// CHECK-SAME: %arg0 : !modelica.int
// CHECK: %[[size:[a-zA-Z0-9]*]] = modelica.cast %arg0 : !modelica.int -> index
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: %[[size]]
// CHECK-SAME: name = "y"
// CHECK-SAME: index -> !modelica.member<heap, ?x!modelica.real>

function sizeDependingOnIntegerInput
    input Integer n;
    output Real[n] y;

algorithm
end sizeDependingOnIntegerInput;
