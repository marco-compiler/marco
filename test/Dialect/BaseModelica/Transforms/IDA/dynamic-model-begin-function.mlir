// RUN: marco-opt %s --split-input-file --ida | FileCheck %s

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           ida.create @ida_main
// CHECK-NEXT:  }

bmodelica.model @emptyModel {

}

// -----

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           ida.create @ida_main
// CHECK-NEXT:      %[[buffer:.*]] = memref.alloc() : memref<1x!ida.variable>
// CHECK-NEXT:      func.call @[[add_variables:.*]](%[[buffer]]) : (memref<1x!ida.variable>) -> ()
// CHECK-NEXT:  }

// CHECK:       func.func private @[[add_variables]](%[[arg:.*]]: memref<1x!ida.variable>) {
// CHECK-NEXT:      %[[state_ida:.*]] = ida.add_state_variable @ida_main {derivativeGetter = @{{.*}}, derivativeSetter = @{{.*}}, dimensions = [1], stateGetter = @{{.*}}, stateSetter = @{{.*}}}
// CHECK-NEXT:      %[[index:.*]] = arith.constant 0 : index
// CHECK-NEXT:      memref.store %[[state_ida]], %[[arg]][%[[index]]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.model @variables der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}
