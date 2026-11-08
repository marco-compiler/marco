// RUN: marco-opt %s --split-input-file --modelica-instantiate=model=M --verify-diagnostics

// expected-error @+1 {{unknown class 'M'}}
module {
    modelica.class model @N {
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{unknown class '@Missing' of component 'x'}}
    modelica.component @x : @Missing -> !modelica.object<@Missing>
}

// -----

modelica.class model @M {
    modelica.component @x : @Real -> !modelica.real {
        // expected-error @+1 {{unknown name 'y'}}
        %0 = modelica.ref @y : !modelica.real
        modelica.bind %0 : !modelica.real
    }
}

// -----

modelica.class model @M {
    modelica.component @n : @Integer -> !modelica.int
    // expected-error @+1 {{can't evaluate the size of the dimension 1 of 'x'}}
    modelica.component @x : @Real -> tensor<?x!modelica.real> dims [?] {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

modelica.class connector @Pin {
    modelica.component @v : @Real -> !modelica.real
    modelica.component flow @i : @Real -> !modelica.real
}

modelica.class model @M {
    modelica.component @n : @Integer -> !modelica.int
    modelica.component @p : @Pin -> tensor<2x!modelica.object<@Pin>> dims [2]
    modelica.equations {
        // expected-error @+1 {{the range of a for-equation containing connect-equations must be evaluable}}
        modelica.for_equation %i : !modelica.int in {
            %0 = modelica.constant #modelica<int 1> : !modelica.int
            %1 = modelica.ref @n : !modelica.int
            %2 = modelica.range %0 to %1 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
            modelica.yield %2 : tensor<?x!modelica.int>
        } do {
            modelica.connect {
                %0 = modelica.ref @p : tensor<2x!modelica.object<@Pin>>
                %1 = modelica.subscript %0[%i] : (tensor<2x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
                %2 = modelica.ref @p : tensor<2x!modelica.object<@Pin>>
                %3 = modelica.constant #modelica<int 1> : !modelica.int
                %4 = modelica.subscript %2[%3] : (tensor<2x!modelica.object<@Pin>>, !modelica.int) -> !modelica.object<@Pin>
                modelica.yield %1, %4 : !modelica.object<@Pin>, !modelica.object<@Pin>
            }
        }
    }
}

// -----

modelica.class model @M {
    modelica.component parameter @b : @Boolean -> !modelica.bool {
        %0 = modelica.constant #modelica<bool false> : !modelica.bool
        modelica.bind %0 : !modelica.bool
    }
    modelica.component @x : @Real -> !modelica.real if {
        %0 = modelica.ref @b : !modelica.bool
        modelica.yield %0 : !modelica.bool
    }
    modelica.equations {
        modelica.equation {
            // expected-error @+1 {{reference to the disabled conditional component 'x'}}
            %0 = modelica.ref @x : !modelica.real
            %1 = modelica.constant #modelica<int 0> : !modelica.int
            modelica.equation_sides %0 = %1 : !modelica.real = !modelica.int
        }
    }
}
