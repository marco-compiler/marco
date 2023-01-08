// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// No variables

// CHECK:       simulation.deinit_ic_solvers_function (%[[idaInstance:.*]]: !ida.instance) {
// CHECK-NEXT:      ida.free %[[idaInstance]]
// CHECK-NEXT:      simulation.yield
// CHECK-NEXT:  }

modelica.model @Test {
    modelica.yield
} body {

}
