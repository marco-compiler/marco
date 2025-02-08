// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           ida.create @ida_main
// CHECK-NEXT:  }

bmodelica.model @emptyModel {

}

// -----

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           %[[state_ida:.*]] = ida.add_state_variable @ida_main {derivativeGetter = @{{.*}}, derivativeSetter = @{{.*}}, dimensions = [1], stateGetter = @{{.*}}, stateSetter = @{{.*}}}
// CHECK-NEXT:  }

bmodelica.model @variables der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>
}
