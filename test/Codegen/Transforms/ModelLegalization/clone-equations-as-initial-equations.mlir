// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(legalize-model{model-name=Test})" | FileCheck %s

// CHECK:       ^bb0(%[[x:.*]]: !modelica.array<!modelica.int>):
// CHECK:       modelica.initial_equation {
// CHECK-DAG:       %[[x_load:.*]] = modelica.load %[[x]][]
// CHECK-DAG:       %[[rhs_value:.*]] = modelica.constant #modelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[rhs_value]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.model @Test {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    modelica.yield %0 : !modelica.member<!modelica.int>
} body {
^bb0(%arg0: !modelica.array<!modelica.int>):
    modelica.equation {
        %0 = modelica.load %arg0[] : !modelica.array<!modelica.int>
        %1 = modelica.constant #modelica.int<0>
        %lhs = modelica.equation_side %0 : tuple<!modelica.int>
        %rhs = modelica.equation_side %1 : tuple<!modelica.int>
        modelica.equation_sides %lhs, %rhs : tuple<!modelica.int>, tuple<!modelica.int>
    }
}
