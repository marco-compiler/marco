// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// CHECK:       simulation.function @getIDATime(solvers: [%[[ida:.*]]: !ida.instance], time: [%[[time:.*]]: !modelica.real], variables: [], extra_args: []) -> !modelica.real {
// CHECK-NEXT:      %[[result:.*]] = ida.get_current_time %[[ida]] : !ida.instance -> !modelica.real
// CHECK-NEXT:      simulation.return %[[result]]
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.yield
} body {

}
