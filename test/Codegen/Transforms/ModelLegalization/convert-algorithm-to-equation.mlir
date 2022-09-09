// RUN: modelica-opt %s --split-input-file --pass-pipeline="legalize-model{model-name=Test}" | FileCheck %s

// CHECK:       @Test
// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>, %[[y:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.start (%[[y]] : !modelica.array<!modelica.int>)
// CHECK-NEXT:  %{{.*}} = modelica.constant #modelica.int<[[start_value:.*]]> : !modelica.int

// CHECK:       modelica.equation {
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][]
// CHECK-DAG:       %[[y_load:.*]] = modelica.load %[[y]][]
// CHECK:           %[[res:.*]] = modelica.call @Test_algorithm_0(%[[x_load]]) : (!modelica.int) -> !modelica.int
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[y_load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[res]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

// CHECK:       @Test_algorithm_0
// CHECK-SAME:  (!modelica.int) -> !modelica.int
// CHECK-DAG:   %[[output:.*]] = modelica.member_create @{{.*}} : !modelica.member<!modelica.int, output>
// CHECK-DAG:   %[[input:.*]] = modelica.member_create @{{.*}} : !modelica.member<!modelica.int, input>
// CHECK:       %[[output_start:.*]] = modelica.constant #modelica.int<[[start_value]]>
// CHECK:       modelica.member_store %[[output]], %[[output_start]]
// CHECK:       %[[input_value:.*]] = modelica.member_load %[[input]]
// CHECK:       modelica.member_store %[[output]], %[[input_value]]

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_create @y : !modelica.member<!modelica.int>
    modelica.yield %0, %1 : !modelica.member<!modelica.int>, !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>, %arg1: !modelica.array<!modelica.int>):
    modelica.start (%arg1 : !modelica.array<!modelica.int>) {each = false, fixed = false} {
        %0 = modelica.constant #modelica.int<57> : !modelica.int
        modelica.yield %0 : !modelica.int
    }
    modelica.algorithm {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        modelica.store %arg1[], %0 : !modelica.array<!modelica.int>
    }
}