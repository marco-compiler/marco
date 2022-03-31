// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @arrayCopy
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "x"
// CHECK-SAME: !modelica.member<?x!modelica.int, input>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
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
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[CONST:[a-zA-Z0-9]*]] = modelica.constant #modelica.int<10>
// CHECK: modelica.member_store %[[Y]], %[[CONST]]

function constantOutput
    output Integer y;

algorithm
    y := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "x"
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<!modelica.real, output>
// CHECK: %[[X_VALUE:[a-zA-Z0-9]*]] = modelica.member_load %[[X]]
// CHECK: modelica.member_store %[[Y]], %[[X_VALUE]]

function castIntegerToReal
    input Integer x;
    output Real y;

algorithm
    y := x;
end castIntegerToReal;


// CHECK-LABEL: @sizeDependingOnIntegerInput
// CHECK: %[[X:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: name = "n"
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[X_VALUE:[a-zA-Z0-9]*]] = modelica.member_load %[[X]]
// CHECK: %[[size:[a-zA-Z0-9]*]] = modelica.cast %[[X_VALUE]] : !modelica.int -> index
// CHECK: %[[Y:[a-zA-Z0-9]*]] = modelica.member_create
// CHECK-SAME: %[[size]]
// CHECK-SAME: name = "y"
// CHECK-SAME: !modelica.member<?x!modelica.real, output>

function sizeDependingOnIntegerInput
    input Integer n;
    output Real[n] y;

algorithm
end sizeDependingOnIntegerInput;
