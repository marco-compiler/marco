// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// Empty function.

// CHECK:       func.func @deinit() {
// CHECK:           return
// CHECK-NEXT:  }

runtime.deinit_function {
    runtime.yield
}

// -----

// Non-empty function.

// CHECK:       func.func @deinit() {
// CHECK:           %[[cst:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.print %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

runtime.deinit_function {
    %0 = modelica.constant #modelica.int<0>
    modelica.print %0 : !modelica.int
    runtime.yield
}
