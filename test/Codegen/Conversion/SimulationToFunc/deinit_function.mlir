// RUN: modelica-opt %s --split-input-file --convert-simulation-to-func | FileCheck %s

// Empty function.

// CHECK:       func.func @deinit() {
// CHECK:           return
// CHECK-NEXT:  }

simulation.deinit_function {
    simulation.yield
}

// -----

// Non-empty function.

// CHECK:       func.func @deinit() {
// CHECK:           %[[cst:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.print %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

simulation.deinit_function {
    %0 = modelica.constant #modelica.int<0>
    modelica.print %0 : !modelica.int
    simulation.yield
}
