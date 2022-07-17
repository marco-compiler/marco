// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @variableCopy
// CHECK: %[[x:.*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[y:.*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[x_value:.*]] = modelica.member_load %[[x]]
// CHECK: modelica.member_store %[[y]], %[[x_value]]

function variableCopy
    input Integer x;
    output Integer y;
algorithm
    y := x;
end variableCopy;


// CHECK-LABEL: @arrayCopy
// CHECK: %[[x:.*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<?x!modelica.int, input>
// CHECK: %[[y:.*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<?x!modelica.int, output>
// CHECK: %[[x_value:.*]] = modelica.member_load %[[x]]
// CHECK: modelica.member_store %[[y]], %[[x_value]]

function arrayCopy
    input Integer[:] x;
    output Integer[:] y;
algorithm
    y := x;
end arrayCopy;


// CHECK-LABEL: @constantOutput
// CHECK: %[[y:.*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.int, output>
// CHECK: %[[const:.*]] = modelica.constant #modelica.int<10>
// CHECK: modelica.member_store %[[y]], %[[const]]

function constantOutput
    output Integer y;
algorithm
    y := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK: %[[x:.*]] = modelica.member_create @x
// CHECK-SAME: !modelica.member<!modelica.int, input>
// CHECK: %[[y:.*]] = modelica.member_create @y
// CHECK-SAME: !modelica.member<!modelica.real, output>
// CHECK: %[[x_value:.*]] = modelica.member_load %[[x]]
// CHECK: modelica.member_store %[[y]], %[[x_value]]

function castIntegerToReal
    input Integer x;
    output Real y;
algorithm
    y := x;
end castIntegerToReal;
