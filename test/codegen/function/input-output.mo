// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

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


