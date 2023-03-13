// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// No variables

// CHECK:       simulation.init_main_solvers_function () -> !ida.instance {
// CHECK-NEXT:      %[[idaInstance:.*]] = ida.create
// CHECK-NEXT:      ida.init %[[idaInstance]]
// CHECK-NEXT:      simulation.yield %[[idaInstance]]
// CHECK-NEXT:  }

modelica.model @Test {

}
