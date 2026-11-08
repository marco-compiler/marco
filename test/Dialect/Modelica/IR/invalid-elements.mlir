// RUN: marco-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{components must be declared within classes}}
modelica.component @x : @Real -> !modelica.real

// -----

// expected-error @+1 {{top-level definitions can't be protected}}
modelica.class protected model @M {
}

// -----

modelica.class model @M {
    // expected-note @+1 {{see existing symbol definition here}}
    modelica.component @x : @Real -> !modelica.real
    // expected-error @+1 {{redefinition of symbol named 'x'}}
    modelica.component @x : @Integer -> !modelica.int
}

// -----

modelica.class model @M {
    // expected-error @+1 {{'each' is allowed only within modifications}}
    modelica.component each @x : @Real -> !modelica.real
}

// -----

modelica.class model @M {
    modelica.component @s : @System -> !modelica.object<@System> {
        // expected-error @+1 {{elements declared within modifications must be redeclare or replaceable}}
        modelica.component @medium : @Water -> !modelica.object<@Medium>
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{only replaceable components can have a constraining clause}}
    modelica.component @medium : @Water -> !modelica.object<@Medium> constrainedby @Medium
}

// -----

modelica.class model @M {
    // expected-error @+1 {{dimensions region must yield 2 values, but yields 1}}
    modelica.component @x : @Real -> tensor<?x?x!modelica.real> dims [?, ?] {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{dimensions must be Integer scalars}}
    modelica.component @x : @Real -> tensor<?x!modelica.real> dims [?] {
        %0 = modelica.ref @n : !modelica.real
        modelica.yield %0 : !modelica.real
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{the type '!modelica.real' has less dimensions than the declared ones}}
    modelica.component @x : @Real -> !modelica.real dims [3]
}

// -----

modelica.class model @M {
    modelica.component @x : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<string "a"> : !modelica.string
        // expected-error @+1 {{value of type '!modelica.string' is not compatible with the component type '!modelica.real'}}
        modelica.bind %0 : !modelica.string
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{modification can't have more than one value}}
    modelica.component @x : @Real -> !modelica.real {
        modelica.bind break
        modelica.bind break
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{condition must be a Boolean scalar}}
    modelica.component @x : @Real -> !modelica.real if {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

// expected-error @+1 {{short class definitions must contain exactly one extends-clause}}
modelica.class short type @T {
    modelica.extends @Real
    modelica.extends @Integer
}

// -----

modelica.class package @P {
    // expected-error @+1 {{class-extends redeclarations must be explicitly redeclare}}
    modelica.class replaceable package extends @Medium {
        modelica.extends inherited
    }
}

// -----

modelica.class model @M {
    // expected-error @+1 {{the base class can be omitted only within class-extends redeclarations}}
    modelica.extends inherited
}

// -----

modelica.class model @M {
    // expected-error @+1 {{input / output prefixes and array dimensions are allowed only within short class definitions}}
    modelica.extends @Base dims [3]
}

// -----

modelica.class short model @M {
    modelica.extends @Base {
        // expected-error @+1 {{is not allowed within short class definitions}}
        modelica.deselect @friction
    }
}

// -----

// expected-error @+1 {{only functions can be impure}}
modelica.class impure model @M {
}

// -----

// expected-error @+1 {{only functions can have an external clause}}
modelica.class model @M {
    modelica.external "C"
}

// -----

// expected-error @+1 {{derivative definitions must be functions with an empty body}}
modelica.class function @df der(@f, "x") {
    modelica.component input @x : @Real -> !modelica.real
}

// -----

// expected-error @+1 {{unspecified enumerations can't have literals}}
modelica.enumeration unspecified @E {
    modelica.enum_literal @a
}

// -----

modelica.class package @P {
    // expected-error @+1 {{an import-clause can be only one of renaming, unqualified, or multiple definitions import}}
    "modelica.import"() <{path = @A::@B, alias = @X, isUnqualified}> : () -> ()
}

// -----

modelica.class model @M {
    // expected-error @+1 {{the size of a dimension can't be negative}}
    modelica.component @x : @Real -> tensor<3x!modelica.real> dims [-1]
}

// -----

modelica.class model @M {
    // expected-error @+1 {{'modelica.constant' op is not allowed within the body of a class}}
    %0 = modelica.constant #modelica<int 1> : !modelica.int
}
