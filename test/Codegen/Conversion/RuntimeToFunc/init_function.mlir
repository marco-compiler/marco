// RUN: modelica-opt %s --split-input-file --convert-runtime-to-func | FileCheck %s

// Empty function.

// CHECK:       func.func @init() {
// CHECK:           return
// CHECK-NEXT:  }

runtime.init_function {
    runtime.yield
}

// -----

// Non-empty function.

// CHECK:       func.func @init() {
// CHECK:           %[[cst:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK:           bmodelica.print %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

runtime.init_function {
    %0 = bmodelica.constant #bmodelica<int 0>
    bmodelica.print %0 : !bmodelica.int
    runtime.yield
}
