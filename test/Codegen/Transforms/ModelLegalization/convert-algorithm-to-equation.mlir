// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test debug-view=true})" | FileCheck %s

// CHECK:       @Test
// CHECK:       modelica.start @y
// CHECK-NEXT:  %{{.*}} = modelica.constant #modelica.int<[[start_value:.*]]> : !modelica.int

// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @y
// CHECK:           %[[res:.*]] = modelica.call @Test_algorithm_0(%[[x]]) : (!modelica.int) -> !modelica.int
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[y]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[res]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

// CHECK:       @Test_algorithm_0
// CHECK-DAG:   modelica.variable @x : !modelica.member<!modelica.int, input>
// CHECK-DAG:   modelica.variable @y : !modelica.member<!modelica.int, output>
// CHECK:       modelica.default @y {
// CHECK-NEXT:      %[[output_start:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      modelica.yield %[[output_start]]
// CHECK-NEXT:  }
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %0 = modelica.variable_get @x
// CHECK-NEXT:      modelica.variable_set @y, %0
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.int>
    modelica.variable @y : !modelica.member<!modelica.int>

    modelica.start @y {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = false, fixed = false}

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        modelica.variable_set @y, %0 : !modelica.int
    }
}

// -----

// Derivative inside algorithm.

// CHECK:       @Test
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:           %[[res:.*]] = modelica.call @Test_algorithm_0(%[[der_x]]) : (!modelica.real) -> !modelica.real
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[res]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

// CHECK:       @Test_algorithm_0
// CHECK-DAG:   modelica.variable @der_x : !modelica.member<!modelica.real, input>
// CHECK-DAG:   modelica.variable @x : !modelica.member<!modelica.real, output>
// CHECK:       modelica.default @x {
// CHECK-NEXT:      %0 = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      modelica.yield %0
// CHECK-NEXT:  }
// CHECK:       %[[x:.*]] = modelica.variable_get @der_x
// CHECK:       modelica.variable_set @x, %[[x]]

modelica.model @Test {
    modelica.variable @x : !modelica.member<!modelica.real>

    modelica.start @x {
        %0 = modelica.constant #modelica.real<0.0> : !modelica.real
        modelica.yield %0 : !modelica.real
    } {each = false, fixed = false}

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.der %0 : !modelica.real -> !modelica.real
        modelica.variable_set @x, %1 : !modelica.real
    }
}
