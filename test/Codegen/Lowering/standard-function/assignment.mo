// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @variableCopy
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<!bmodelica.int, output>
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.int
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function variableCopy
    input Integer x;
    output Integer y;
algorithm
    y := x;
end variableCopy;


// CHECK-LABEL: @arrayCopy
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<?x!bmodelica.int, input>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<?x!bmodelica.int, output>
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.array<?x!bmodelica.int>
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function arrayCopy
    input Integer[:] x;
    output Integer[:] y;
algorithm
    y := x;
end arrayCopy;


// CHECK-LABEL: @constantOutput
// CHECK: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, output>
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[const:.*]] = bmodelica.constant #bmodelica.int<10>
// CHECK-NEXT:      bmodelica.variable_set @x, %[[const]]
// CHECK-NEXT:  }

function constantOutput
    output Integer x;
algorithm
    x := 10;
end constantOutput;


// CHECK-LABEL: @castIntegerToReal
// CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
// CHECK-DAG: bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK:       bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.int
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:  }

function castIntegerToReal
    input Integer x;
    output Real y;
algorithm
    y := x;
end castIntegerToReal;
