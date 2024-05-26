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
// CHECK:           %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK:           bmodelica.print %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

runtime.deinit_function {
    %0 = bmodelica.constant #bmodelica<int 0>
    bmodelica.print %0 : !bmodelica.int
    runtime.yield
}
