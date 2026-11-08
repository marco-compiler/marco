// RUN: marco-opt %s --split-input-file | FileCheck %s

// Statements.

// CHECK-LABEL: modelica.class function @Assignments
// CHECK:       modelica.assignment {
// CHECK:           modelica.assignment_sides %{{.*}} = %{{.*}} : !modelica.real = !modelica.int
// CHECK-NEXT:  }
// CHECK:       modelica.assignment {
// CHECK:           modelica.assignment_sides %{{.*}}, %{{.*}} = %{{.*}}#0, %{{.*}}#1 : !modelica.real, !modelica.int = !modelica.real, !modelica.int
// CHECK-NEXT:  }
// CHECK:       modelica.assignment {
// CHECK:           %[[elements:.*]] = modelica.subscript %{{.*}}[%{{.*}}] : (tensor<5x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x!modelica.real>
// CHECK:           modelica.assignment_sides %[[elements]] = %{{.*}} : tensor<2x!modelica.real> = tensor<2x!modelica.int>
// CHECK-NEXT:  }
// CHECK:       modelica.call_statement @print {

modelica.class function @Assignments {
    modelica.algorithm {
        // x := 1
        modelica.assignment {
            %x = modelica.ref @x : !modelica.real
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            modelica.assignment_sides %x = %0 : !modelica.real = !modelica.int
        }

        // (a, b) := f()
        modelica.assignment {
            %a = modelica.ref @a : !modelica.real
            %b = modelica.ref @b : !modelica.int
            %0:2 = modelica.call @f() : () -> (!modelica.real, !modelica.int)
            modelica.assignment_sides %a, %b = %0#0, %0#1 : !modelica.real, !modelica.int = !modelica.real, !modelica.int
        }

        // v[{1, 3}] := {2, 3}
        modelica.assignment {
            %v = modelica.ref @v : tensor<5x!modelica.real>
            %0 = modelica.constant #modelica<array [1, 3]> : tensor<2x!modelica.int>
            %1 = modelica.subscript %v[%0] : (tensor<5x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x!modelica.real>
            %2 = modelica.constant #modelica<array [2, 3]> : tensor<2x!modelica.int>
            modelica.assignment_sides %1 = %2 : tensor<2x!modelica.real> = tensor<2x!modelica.int>
        }

        modelica.call_statement @print {
            %0 = modelica.constant #modelica<string "hello"> : !modelica.string
            modelica.yield %0 : !modelica.string
        }
    }
}

// -----

// Assignment of a derivative.

// CHECK-LABEL: modelica.class model @DerivativeAssignment
// CHECK:       %[[der:.*]] = modelica.der
// CHECK:       modelica.assignment_sides %[[der]] = %{{.*}}

modelica.class model @DerivativeAssignment {
    modelica.algorithm {
        modelica.assignment {
            %x = modelica.ref @x : !modelica.real
            %0 = modelica.der %x : (!modelica.real) -> !modelica.real
            %1 = modelica.constant #modelica<real 1.0> : !modelica.real
            modelica.assignment_sides %0 = %1 : !modelica.real = !modelica.real
        }
    }
}

// -----

// Control flow statements.

// CHECK-LABEL: modelica.class function @ControlFlow
// CHECK:       modelica.for %[[i:.*]] : !modelica.int in {
// CHECK:       } do {
// CHECK-NEXT:      modelica.if {
// CHECK:           } then {
// CHECK-NEXT:          modelica.break
// CHECK-NEXT:      } else {
// CHECK-NEXT:          modelica.return
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK:       modelica.while {
// CHECK:       } do {
// CHECK-NEXT:      modelica.assert {

modelica.class function @ControlFlow {
    modelica.algorithm {
        modelica.for %i : !modelica.int in {
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            %1 = modelica.constant #modelica<int 10> : !modelica.int
            %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<10x!modelica.int>
            modelica.yield %2 : tensor<10x!modelica.int>
        } do {
            modelica.if {
                %0 = modelica.constant #modelica<int 5> : !modelica.int
                %1 = modelica.gt %i, %0 : (!modelica.int, !modelica.int) -> !modelica.bool
                modelica.yield %1 : !modelica.bool
            } then {
                modelica.break
            } else {
                modelica.return
            }
        }

        modelica.while {
            %c = modelica.ref @c : !modelica.bool
            modelica.yield %c : !modelica.bool
        } do {
            modelica.assert {
                %c = modelica.ref @c : !modelica.bool
                %0 = modelica.constant #modelica<string "message"> : !modelica.string
                modelica.yield %c, %0 : !modelica.bool, !modelica.string
            }
        }
    }
}

// -----

// When-statements.

// CHECK-LABEL: modelica.class model @WhenStatements
// CHECK:       modelica.when {
// CHECK:       } then {
// CHECK:       } elsewhen {
// CHECK-NEXT:      modelica.when {

modelica.class model @WhenStatements {
    modelica.algorithm {
        modelica.when {
            %start = modelica.constant #modelica<int 0> : !modelica.int
            %interval = modelica.constant #modelica<real 0.1> : !modelica.real
            %0 = modelica.sample %start, %interval : (!modelica.int, !modelica.real) -> !modelica.bool
            modelica.yield %0 : !modelica.bool
        } then {
            modelica.assignment {
                %x = modelica.ref @x : !modelica.real
                %0 = modelica.constant #modelica<real 1.0> : !modelica.real
                modelica.assignment_sides %x = %0 : !modelica.real = !modelica.real
            }
        } elsewhen {
            modelica.when {
                %0 = modelica.initial : !modelica.bool
                modelica.yield %0 : !modelica.bool
            } then {
            }
        }
    }
}
