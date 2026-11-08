// RUN: marco-opt %s --split-input-file --modelica-instantiate=model=M --verify-diagnostics | FileCheck %s

// Outer components refer to the inner components of the enclosing
// instances.

// CHECK-LABEL: modelica.model @M {
// CHECK:           modelica.instance model @world : !modelica.object<@World> {
// CHECK-NEXT:          modelica.variable parameter @g : !modelica.real {
// CHECK-NEXT:              modelica.constant #modelica<int 10>
// CHECK:           modelica.instance model @b : !modelica.object<@Body> {
// CHECK-NEXT:          modelica.variable @f : !modelica.real {
// CHECK-NEXT:              %[[world:.*]] = modelica.ref @world : !modelica.object<@World>
// CHECK-NEXT:              %[[g:.*]] = modelica.member %[[world]], @g
// CHECK-NEXT:              modelica.bind %[[g]]
// CHECK-NEXT:          }
// CHECK-NEXT:      }

modelica.class model @World {
    modelica.component parameter @g : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<real 9.81> : !modelica.real
        modelica.bind %0 : !modelica.real
    }
}

modelica.class model @Body {
    modelica.component outer @world : @World -> !modelica.object<@World>
    modelica.component @f : @Real -> !modelica.real {
        %0 = modelica.ref @world : !modelica.object<@World>
        %1 = modelica.member %0, @g : (!modelica.object<@World>) -> !modelica.real
        modelica.bind %1 : !modelica.real
    }
}

modelica.class model @M {
    modelica.component inner @world : @World -> !modelica.object<@World> {
        modelica.modify @g {
            %0 = modelica.constant #modelica<int 10> : !modelica.int
            modelica.bind %0 : !modelica.int
        }
    }
    modelica.component @b : @Body -> !modelica.object<@Body>
}

// -----

// Missing inner components are added at the top level.

// CHECK-LABEL: modelica.model @M {
// CHECK:           modelica.instance model @world : !modelica.object<@World> {
// CHECK-NEXT:          modelica.variable parameter @g : !modelica.real {
// CHECK-NEXT:              modelica.constant #modelica<real 9.810000e+00>
// CHECK:           modelica.instance model @b : !modelica.object<@Body> {
// CHECK:               modelica.ref @world

modelica.class model @World {
    modelica.component parameter @g : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<real 9.81> : !modelica.real
        modelica.bind %0 : !modelica.real
    }
}

modelica.class model @Body {
    // expected-warning @+1 {{no inner declaration for the outer component 'b.world', adding one at the top level}}
    modelica.component outer @world : @World -> !modelica.object<@World>
    modelica.component @f : @Real -> !modelica.real {
        %0 = modelica.ref @world : !modelica.object<@World>
        %1 = modelica.member %0, @g : (!modelica.object<@World>) -> !modelica.real
        modelica.bind %1 : !modelica.real
    }
}

modelica.class model @M {
    modelica.component @b : @Body -> !modelica.object<@Body>
}
