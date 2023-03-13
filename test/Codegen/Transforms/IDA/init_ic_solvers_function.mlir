// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(ida{model=Test})" | FileCheck %s

// No variables

// CHECK:       simulation.init_ic_solvers_function () -> !ida.instance {
// CHECK-NEXT:      %[[idaInstance:.*]] = ida.create
// CHECK-DAG:       ida.set_start_time %[[idaInstance]] {time = 0.000000e+00 : f64}
// CHECK-DAG:       ida.set_end_time %[[idaInstance]] {time = 0.000000e+00 : f64}
// CHECK-NEXT:      ida.init %[[idaInstance]]
// CHECK-NEXT:      simulation.yield %[[idaInstance]]
// CHECK-NEXT:  }

modelica.model @Test {

}
