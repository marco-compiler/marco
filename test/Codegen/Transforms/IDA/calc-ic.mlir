// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// CHECK:       simulation.function @calcIC(solvers: [%[[ida:.*]]: !ida.instance], time: [%[[time:.*]]: !modelica.real], variables: [], extra_args: []) -> () {
// CHECK-NEXT:      ida.calc_ic %[[ida]]
// CHECK-NEXT:      simulation.return
// CHECK-NEXT:  }

modelica.model @Test {

}
