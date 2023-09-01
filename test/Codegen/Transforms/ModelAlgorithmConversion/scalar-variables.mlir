// RUN: modelica-opt %s --split-input-file --convert-model-algorithms | FileCheck %s

// CHECK:       @Test
// CHECK:       modelica.start @y {
// CHECK:           %[[start:.*]] = modelica.constant #modelica.int<[[start_value:.*]]> : !modelica.int
// CHECK:           modelica.yield %[[start]]
// CHECK-NEXT:  }
// CHECK:       %[[t0:.*]] = modelica.equation_template inductions = [] {
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @x
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @y
// CHECK:           %[[res:.*]] = modelica.call @Test_algorithm_0(%[[x]]) : (!modelica.int) -> !modelica.int
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[y]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[res]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       modelica.equation_instance %[[t0]]

// CHECK:       @Test_algorithm_0
// CHECK-DAG:   modelica.variable @x : !modelica.variable<!modelica.int, input>
// CHECK-DAG:   modelica.variable @y : !modelica.variable<!modelica.int, output>
// CHECK:       modelica.default @y {
// CHECK-NEXT:      %[[default:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      modelica.yield %[[default]]
// CHECK-NEXT:  }
// CHECK:       modelica.algorithm {
// CHECK-NEXT:      %0 = modelica.variable_get @x
// CHECK-NEXT:      modelica.variable_set @y, %0
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.int>
    modelica.variable @y : !modelica.variable<!modelica.int>

    modelica.start @y {
        %0 = modelica.constant #modelica.int<0> : !modelica.int
        modelica.yield %0 : !modelica.int
    } {each = false, fixed = false}

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.int
        modelica.variable_set @y, %0 : !modelica.int
    }
}
