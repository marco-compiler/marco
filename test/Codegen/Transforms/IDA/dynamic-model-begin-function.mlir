// RUN: modelica-opt %s --split-input-file --ida | FileCheck %s

// Empty model.

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           ida.create @ida_main
// CHECK-NEXT:  }

modelica.model @Test {

}

// -----

// Check variables.

// CHECK: ida.instance @ida_main
// CHECK:       runtime.dynamic_model_begin {
// CHECK:           %[[state_ida:.*]] = ida.add_state_variable @ida_main {derivativeGetter = @{{.*}}, derivativeSetter = @{{.*}}, dimensions = [1], stateGetter = @{{.*}}, stateSetter = @{{.*}}}
// CHECK-NEXT:  }

modelica.model @Test attributes {derivatives_map = [#modelica<var_derivative @x, @der_x>]} {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @der_x : !modelica.variable<!modelica.real>
}
