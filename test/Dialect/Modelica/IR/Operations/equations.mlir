// RUN: marco-opt %s --split-input-file | FileCheck %s

// Equations.

// CHECK-LABEL: modelica.class model @SimpleEquations
// CHECK:       modelica.equation {
// CHECK:           modelica.equation_sides %{{.*}} = %{{.*}} : !modelica.real = !modelica.int
// CHECK-NEXT:  }
// CHECK:       modelica.equation {
// CHECK:           %[[results:.*]]:3 = modelica.call @f(%{{.*}}) : (!modelica.real) -> (!modelica.real, !modelica.int, !modelica.real)
// CHECK:           modelica.equation_sides %{{.*}}, %{{.*}} = %[[results]]#0, %[[results]]#2 : !modelica.real, !modelica.real = !modelica.real, !modelica.real
// CHECK-NEXT:  }
// CHECK:       modelica.call_equation @print {
// CHECK:           modelica.yield %{{.*}} : !modelica.string
// CHECK-NEXT:  }

modelica.class model @SimpleEquations {
    modelica.equations {
        // x = 1
        modelica.equation {
            %x = modelica.ref @x : !modelica.real
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            modelica.equation_sides %x = %0 : !modelica.real = !modelica.int
        }

        // (a, , b) = f(x)
        modelica.equation {
            %a = modelica.ref @a : !modelica.real
            %b = modelica.ref @b : !modelica.real
            %x = modelica.ref @x : !modelica.real
            %0:3 = modelica.call @f(%x) : (!modelica.real) -> (!modelica.real, !modelica.int, !modelica.real)
            modelica.equation_sides %a, %b = %0#0, %0#2 : !modelica.real, !modelica.real = !modelica.real, !modelica.real
        }

        // print("hello")
        modelica.call_equation @print {
            %0 = modelica.constant #modelica<string "hello"> : !modelica.string
            modelica.yield %0 : !modelica.string
        }
    }
}

// -----

// For-equations, with explicit and implicit ranges.

// CHECK-LABEL: modelica.class model @ForEquations
// CHECK:       modelica.for_equation %[[i:.*]] : !modelica.int in {
// CHECK:           modelica.yield %{{.*}} : tensor<?x!modelica.int>
// CHECK-NEXT:  } do {
// CHECK-NEXT:      modelica.for_equation %[[j:.*]] : !modelica.real in {
// CHECK:           } do {
// CHECK-NEXT:          modelica.equation {
// CHECK:       modelica.for_equation %{{.*}} : !modelica.int implicit do {

modelica.class model @ForEquations {
    modelica.equations {
        modelica.for_equation %i : !modelica.int in {
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            %n = modelica.ref @n : !modelica.int
            %1 = modelica.range %0 to %n : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
            modelica.yield %1 : tensor<?x!modelica.int>
        } do {
            modelica.for_equation %j : !modelica.real in {
                %0 = modelica.constant #modelica<array [1.0, 2.5]> : tensor<2x!modelica.real>
                modelica.yield %0 : tensor<2x!modelica.real>
            } do {
                modelica.equation {
                    %x = modelica.ref @x : tensor<?x!modelica.real>
                    %xi = modelica.subscript %x[%i] : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.real
                    modelica.equation_sides %xi = %j : !modelica.real = !modelica.real
                }
            }
        }

        modelica.for_equation %i : !modelica.int implicit do {
        }
    }
}

// -----

// If-equations and when-equations.

// CHECK-LABEL: modelica.class model @ConditionalEquations
// CHECK:       modelica.if_equation {
// CHECK:       } then {
// CHECK:       } else {
// CHECK-NEXT:      modelica.if_equation {
// CHECK:           } then {
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK:       modelica.when_equation {
// CHECK:           modelica.yield %{{.*}} : tensor<2x!modelica.bool>
// CHECK-NEXT:  } then {
// CHECK:           modelica.reinit {
// CHECK:       } elsewhen {
// CHECK-NEXT:      modelica.when_equation {
// CHECK:           } then {
// CHECK:               modelica.terminate {

modelica.class model @ConditionalEquations {
    modelica.equations {
        modelica.if_equation {
            %c = modelica.ref @c : !modelica.bool
            modelica.yield %c : !modelica.bool
        } then {
        } else {
            modelica.if_equation {
                %d = modelica.ref @d : !modelica.bool
                modelica.yield %d : !modelica.bool
            } then {
            }
        }

        modelica.when_equation {
            %c = modelica.ref @c : !modelica.bool
            %d = modelica.ref @d : !modelica.bool
            %0 = modelica.array %c, %d : (!modelica.bool, !modelica.bool) -> tensor<2x!modelica.bool>
            modelica.yield %0 : tensor<2x!modelica.bool>
        } then {
            modelica.reinit {
                %x = modelica.ref @x : !modelica.real
                %0 = modelica.constant #modelica<int 0> : !modelica.int
                modelica.yield %x, %0 : !modelica.real, !modelica.int
            }
        } elsewhen {
            modelica.when_equation {
                %e = modelica.ref @e : !modelica.bool
                modelica.yield %e : !modelica.bool
            } then {
                modelica.terminate {
                    %0 = modelica.constant #modelica<string "done"> : !modelica.string
                    modelica.yield %0 : !modelica.string
                }
            }
        }
    }
}

// -----

// Assertions.

// CHECK-LABEL: modelica.class model @Assertions
// CHECK:       modelica.assert {
// CHECK:           modelica.yield %{{.*}}, %{{.*}} : !modelica.bool, !modelica.string
// CHECK:       modelica.assert {
// CHECK:           modelica.yield %{{.*}}, %{{.*}}, %{{.*}} : !modelica.bool, !modelica.string, !modelica.enum<@AssertionLevel>

modelica.class model @Assertions {
    modelica.equations {
        modelica.assert {
            %c = modelica.ref @c : !modelica.bool
            %0 = modelica.constant #modelica<string "failure"> : !modelica.string
            modelica.yield %c, %0 : !modelica.bool, !modelica.string
        }

        modelica.assert {
            %c = modelica.ref @c : !modelica.bool
            %0 = modelica.constant #modelica<string "warning"> : !modelica.string
            %1 = modelica.constant #modelica<enum "warning" = 1> : !modelica.enum<@AssertionLevel>
            modelica.yield %c, %0, %1 : !modelica.bool, !modelica.string, !modelica.enum<@AssertionLevel>
        }
    }
}

// -----

// Connections.

// CHECK-LABEL: modelica.class model @Connections
// CHECK:       modelica.connect {
// CHECK:           %[[rn:.*]] = modelica.member %{{.*}}, @n : (!modelica.object<@Resistor>) -> !modelica.object<@Pin>
// CHECK:           %[[p:.*]] = modelica.ref @p : !modelica.object<@Pin>
// CHECK:           modelica.yield %[[rn]], %[[p]] : !modelica.object<@Pin>, !modelica.object<@Pin>
// CHECK-NEXT:  }
// CHECK:       modelica.connections_branch {
// CHECK:       modelica.connections_root {
// CHECK:       modelica.connections_potential_root {
// CHECK:       modelica.connections_unique_root {

modelica.class model @Connections {
    modelica.equations {
        modelica.connect {
            %r = modelica.ref @r : !modelica.object<@Resistor>
            %0 = modelica.member %r, @n : (!modelica.object<@Resistor>) -> !modelica.object<@Pin>
            %p = modelica.ref @p : !modelica.object<@Pin>
            modelica.yield %0, %p : !modelica.object<@Pin>, !modelica.object<@Pin>
        }

        modelica.connections_branch {
            %a = modelica.ref @a : !modelica.object<@Frame>
            %aR = modelica.member %a, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
            %b = modelica.ref @b : !modelica.object<@Frame>
            %bR = modelica.member %b, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
            modelica.yield %aR, %bR : !modelica.object<@Orientation>, !modelica.object<@Orientation>
        }

        modelica.connections_root {
            %a = modelica.ref @a : !modelica.object<@Frame>
            %aR = modelica.member %a, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
            modelica.yield %aR : !modelica.object<@Orientation>
        }

        modelica.connections_potential_root {
            %a = modelica.ref @a : !modelica.object<@Frame>
            %aR = modelica.member %a, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            modelica.yield %aR, %0 : !modelica.object<@Orientation>, !modelica.int
        }

        modelica.connections_unique_root {
            %a = modelica.ref @a : !modelica.object<@Frame>
            %aR = modelica.member %a, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
            %0 = modelica.constant #modelica<string "not unique"> : !modelica.string
            modelica.yield %aR, %0 : !modelica.object<@Orientation>, !modelica.string
        }
    }
}
