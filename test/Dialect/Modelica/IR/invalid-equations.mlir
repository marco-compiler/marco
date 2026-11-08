// RUN: marco-opt %s --split-input-file --verify-diagnostics

modelica.class model @M {
    modelica.equations {
        modelica.equation {
            %x = modelica.ref @x : !modelica.real
            // expected-error @+1 {{the two sides must have the same non-zero number of values}}
            modelica.equation_sides %x, %x = %x : !modelica.real, !modelica.real = !modelica.real
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        modelica.equation {
            %x = modelica.ref @x : !modelica.real
            %0 = modelica.constant #modelica<string "a"> : !modelica.string
            // expected-error @+1 {{incompatible types '!modelica.real' and '!modelica.string'}}
            modelica.equation_sides %x = %0 : !modelica.real = !modelica.string
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{connected element must be a component reference}}
        modelica.connect {
            %0 = modelica.ref @a : !modelica.object<@Pin>
            %1 = modelica.call @makePin() : () -> !modelica.object<@Pin>
            modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{body region must yield 2 values, but yields 1}}
        modelica.connect {
            %0 = modelica.ref @a : !modelica.object<@Pin>
            modelica.yield %0 : !modelica.object<@Pin>
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{reinitialized variable must be Real}}
        modelica.reinit {
            %0 = modelica.ref @n : !modelica.int
            modelica.yield %0, %0 : !modelica.int, !modelica.int
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{message must be a String scalar}}
        modelica.assert {
            %0 = modelica.ref @c : !modelica.bool
            modelica.yield %0, %0 : !modelica.bool, !modelica.bool
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{loop variable type '!modelica.real' doesn't match the range type 'tensor<3x!modelica.int>'}}
        modelica.for_equation %i : !modelica.real in {
            %0 = modelica.constant #modelica<array [1, 2, 3]> : tensor<3x!modelica.int>
            modelica.yield %0 : tensor<3x!modelica.int>
        } do {
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{condition must be a Boolean scalar, but got '!modelica.int'}}
        modelica.if_equation {
            %0 = modelica.ref @n : !modelica.int
            modelica.yield %0 : !modelica.int
        } then {
        }
    }
}

// -----

// When-equations can't be nested.

modelica.class model @M {
    modelica.equations {
        modelica.when_equation {
            %0 = modelica.ref @c : !modelica.bool
            modelica.yield %0 : !modelica.bool
        } then {
            // expected-error @+1 {{when-equations can't be nested}}
            modelica.when_equation {
                %0 = modelica.ref @d : !modelica.bool
                modelica.yield %0 : !modelica.bool
            } then {
            }
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations initial {
        // expected-error @+1 {{when-equations can't be used within initial equations}}
        modelica.when_equation {
            %0 = modelica.ref @c : !modelica.bool
            modelica.yield %0 : !modelica.bool
        } then {
        }
    }
}

// -----

modelica.class model @M {
    modelica.equations {
        // expected-error @+1 {{elsewhen region must contain exactly one 'modelica.when_equation'}}
        modelica.when_equation {
            %0 = modelica.ref @c : !modelica.bool
            modelica.yield %0 : !modelica.bool
        } then {
        } elsewhen {
            modelica.terminate {
                %0 = modelica.constant #modelica<string "end"> : !modelica.string
                modelica.yield %0 : !modelica.string
            }
        }
    }
}

// -----

// When-statements can't be used within loops.

modelica.class model @M {
    modelica.algorithm {
        modelica.while {
            %0 = modelica.ref @c : !modelica.bool
            modelica.yield %0 : !modelica.bool
        } do {
            // expected-error @+1 {{when-statements can't be used within while-loops, for-loops, or if-statements}}
            modelica.when {
                %0 = modelica.ref @d : !modelica.bool
                modelica.yield %0 : !modelica.bool
            } then {
            }
        }
    }
}

// -----

modelica.class function @f {
    modelica.algorithm {
        // expected-error @+1 {{when-statements can't be used within functions}}
        modelica.when {
            %0 = modelica.ref @d : !modelica.bool
            modelica.yield %0 : !modelica.bool
        } then {
        }
    }
}

// -----

modelica.class function @f {
    modelica.algorithm {
        // expected-error @+1 {{must be within a for-loop or a while-loop}}
        modelica.break
    }
}

// -----

modelica.class function @f {
    modelica.algorithm {
        modelica.assignment {
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            // expected-error @+1 {{destination must be a component reference}}
            modelica.assignment_sides %0 = %0 : !modelica.int = !modelica.int
        }
    }
}

// -----

modelica.class function @f {
    modelica.algorithm {
        modelica.assignment {
            %x = modelica.ref @x : !modelica.int
            %0 = modelica.constant #modelica<real 1.5> : !modelica.real
            // expected-error @+1 {{incompatible types '!modelica.int' and '!modelica.real'}}
            modelica.assignment_sides %x = %0 : !modelica.int = !modelica.real
        }
    }
}

// -----

modelica.class model @M {
    modelica.algorithm {
        // expected-error @+1 {{expects parent op to be one of}}
        modelica.equation {
            %x = modelica.ref @x : !modelica.real
            modelica.equation_sides %x = %x : !modelica.real = !modelica.real
        }
    }
}
