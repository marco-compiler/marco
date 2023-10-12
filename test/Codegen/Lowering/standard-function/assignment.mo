// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @variableCopy
// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, input>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.int, output>
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x : !modelica.int
// CHECK-NEXT:      modelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function variableCopy
    input Integer x;
    output Integer y;
algorithm
    y := x;
end variableCopy;


// CHECK-LABEL: @arrayCopy
// CHECK-DAG: modelica.variable @x : !modelica.variable<?x!modelica.int, input>
// CHECK-DAG: modelica.variable @y : !modelica.variable<?x!modelica.int, output>
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x : !modelica.array<?x!modelica.int>
// CHECK-NEXT:      modelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function arrayCopy
    input Integer[:] x;
    output Integer[:] y;
algorithm
    y := x;
end arrayCopy;


// CHECK-LABEL: @constantOutput
// CHECK: modelica.variable @x : !modelica.variable<!modelica.int, output>
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[const:.*]] = modelica.constant #modelica.int<10>
// CHECK-NEXT:      modelica.variable_set @x, %[[const]]
// CHECK-NEXT:  }

function constantOutput
    output Integer x;
algorithm
    x := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK-DAG: modelica.variable @x : !modelica.variable<!modelica.int, input>
// CHECK-DAG: modelica.variable @y : !modelica.variable<!modelica.real, output>
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = modelica.variable_get @x : !modelica.int
// CHECK-NEXT:      modelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function castIntegerToReal
    input Integer x;
    output Real y;
algorithm
    y := x;
end castIntegerToReal;
