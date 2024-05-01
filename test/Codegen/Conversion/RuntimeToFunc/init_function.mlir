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
// CHECK:           %[[cst:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.print %[[cst]]
// CHECK:           return
// CHECK-NEXT:  }

runtime.init_function {
    %0 = modelica.constant #modelica.int<0>
    modelica.print %0 : !modelica.int
    runtime.yield
}
