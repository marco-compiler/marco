// RUN: marco-opt %s --split-input-file --verify-diagnostics

modelica.model @M {
    // expected-error @+1 {{arrays of instances must have a static shape}}
    modelica.instance model @r : tensor<?x!modelica.object<@R>> {
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{invalid instance type '!modelica.real'}}
    modelica.instance model @r : !modelica.real {
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{invalid variable type '!modelica.object<@R>'}}
    modelica.variable @x : !modelica.object<@R>
}

// -----

modelica.model @M {
    // expected-error @+1 {{only the variables of functions can have dynamic dimensions}}
    modelica.variable @x : tensor<?x!modelica.real>
}

// -----

modelica.model @M {
    modelica.instance model @r : tensor<2x!modelica.object<@R>> {
        // expected-error @+1 {{the modification region must have one argument for each dimension of the enclosing instances}}
        modelica.variable @x : !modelica.real {
            %0 = modelica.constant #modelica<real 1.0> : !modelica.real
            modelica.bind %0 : !modelica.real
        }
    }
}

// -----

modelica.model @M {
    modelica.variable @x : !modelica.real {
        %0 = modelica.constant #modelica<string "a"> : !modelica.string
        // expected-error @+1 {{value of type '!modelica.string' is not compatible with the variable type '!modelica.real'}}
        modelica.bind %0 : !modelica.string
    }
}

// -----

modelica.model @M {
    // expected-note @+1 {{see existing symbol definition here}}
    modelica.variable @x : !modelica.real
    // expected-error @+1 {{redefinition of symbol named 'x'}}
    modelica.variable @x : !modelica.real
}

// -----

modelica.function @f {
    // expected-error @+1 {{dimensions region must yield 2 values, but yields 1}}
    modelica.variable output @y : tensor<?x?x!modelica.real> dims {
        %0 = modelica.constant #modelica<int 1> : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}
