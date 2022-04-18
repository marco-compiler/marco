// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @variableCopy
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[X_VALUE:[a-zA-Z0-9]*]] = modelica.member_load %[[X]]
// CHECK: modelica.member_store %[[Y]], %[[X_VALUE]]

function variableCopy
    input Integer x;
    output Integer y;

algorithm
    y := x;
end variableCopy;


// CHECK-LABEL: @arrayCopy
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<?x!modelica.int, input>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<?x!modelica.int, output>
// CHECK: %[[X_VALUE:[a-zA-Z0-9]*]] = modelica.member_load %[[X]]
// CHECK: modelica.member_store %[[Y]], %[[X_VALUE]]

function arrayCopy
    input Integer[:] x;
    output Integer[:] y;

algorithm
    y := x;
end arrayCopy;


// CHECK-LABEL: @constantOutput
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[CONST:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<10>
// CHECK: modelica.member_store %[[Y]], %[[CONST]]

function constantOutput
    output Integer y;

algorithm
    y := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.real, output>
// CHECK: %[[X_VALUE:[a-zA-Z0-9]*]] = modelica.member_load %[[X]]
// CHECK: modelica.member_store %[[Y]], %[[X_VALUE]]

function castIntegerToReal
    input Integer x;
    output Real y;

algorithm
    y := x;
end castIntegerToReal;
