// RUN: marco-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{components must be declared within classes}}
modelica.component @x : @Real -> !modelica.real

// -----

// expected-error @+1 {{top-level definitions can't be protected}}
modelica.model protected @M {
}

// -----

modelica.model @M {
    // expected-note @+1 {{see existing symbol definition here}}
    modelica.component @x : @Real -> !modelica.real
    // expected-error @+1 {{redefinition of symbol named 'x'}}
    modelica.component @x : @Integer -> !modelica.int
}

// -----

modelica.model @M {
    // expected-error @+1 {{'each' is allowed only within modifications}}
    modelica.component each @x : @Real -> !modelica.real
}

// -----

modelica.model @M {
    modelica.component @s : @System -> !modelica.object<@System> {
        // expected-error @+1 {{elements declared within modifications must be redeclare or replaceable}}
        modelica.component @medium : @Water -> !modelica.object<@Medium>
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{only replaceable components can have a constraining clause}}
    modelica.component @medium : @Water -> !modelica.object<@Medium> constrainedby @Medium
}

// -----

modelica.model @M {
    // expected-error @+1 {{dimensions region must yield 2 values, but yields 1}}
    modelica.component @x : @Real -> tensor<?x?x!modelica.real> dims [?, ?] {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{dimensions must be Integer scalars}}
    modelica.component @x : @Real -> tensor<?x!modelica.real> dims [?] {
        %0 = modelica.ref @n : !modelica.real
        modelica.yield %0 : !modelica.real
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{the type '!modelica.real' has less dimensions than the declared ones}}
    modelica.component @x : @Real -> !modelica.real dims [3]
}

// -----

modelica.model @M {
    modelica.component @x : @Real -> !modelica.real {
        %0 = modelica.constant #modelica<string "a"> : !modelica.string
        // expected-error @+1 {{value of type '!modelica.string' is not compatible with the component type '!modelica.real'}}
        modelica.bind %0 : !modelica.string
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{modification can't have more than one value}}
    modelica.component @x : @Real -> !modelica.real {
        modelica.bind break
        modelica.bind break
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{condition must be a Boolean scalar}}
    modelica.component @x : @Real -> !modelica.real if {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

// expected-error @+1 {{short class definitions must contain exactly one extends-clause}}
modelica.type short @T {
    modelica.extends @Real
    modelica.extends @Integer
}

// -----

modelica.package @P {
    // expected-error @+1 {{class-extends redeclarations must be explicitly redeclare}}
    modelica.package replaceable extends @Medium {
        modelica.extends inherited
    }
}

// -----

modelica.model @M {
    // expected-error @+1 {{the base class can be omitted only within class-extends redeclarations}}
    modelica.extends inherited
}

// -----

modelica.model @M {
    // expected-error @+1 {{input / output prefixes and array dimensions are allowed only within short class definitions}}
    modelica.extends @Base dims [3]
}

// -----

modelica.model short @M {
    modelica.extends @Base {
        // expected-error @+1 {{is not allowed within short class definitions}}
        modelica.deselect @friction
    }
}

// -----

// expected-error @+1 {{only functions can have an external clause}}
modelica.model @M {
    modelica.external "C"
}

// -----

// expected-error @+1 {{derivative definitions must have an empty body}}
modelica.function @df der(@f, ["x"]) {
    modelica.component input @x : @Real -> !modelica.real
}

// -----

// expected-error @+1 {{unspecified enumerations can't have literals}}
modelica.enumeration unspecified @E {
    modelica.enum_literal @a
}

// -----

modelica.package @P {
    // expected-error @+1 {{an import-clause can be only one of renaming, unqualified, or multiple definitions import}}
    "modelica.import"() <{path = @A::@B, alias = @X, isUnqualified}> : () -> ()
}

// -----

modelica.model @M {
    // expected-error @+1 {{the size of a dimension can't be negative}}
    modelica.component @x : @Real -> tensor<3x!modelica.real> dims [-1]
}

// -----

modelica.model @M {
    // expected-error @+1 {{'modelica.constant' op is not allowed within the body of a class}}
    %0 = modelica.constant #modelica<int 1> : !modelica.int
}

// -----

modelica.record @R {
    // expected-error @+1 {{is not allowed within records}}
    modelica.equations {
    }
}

// -----

modelica.connector @C {
    // expected-error @+1 {{is not allowed within connectors}}
    modelica.algorithm {
    }
}

// -----

modelica.type @T {
    // expected-error @+1 {{is not allowed within types}}
    modelica.component @x : @Real -> !modelica.real
}

// -----

modelica.package @P {
    // expected-error @+1 {{packages can contain only constant components}}
    modelica.component parameter @x : @Real -> !modelica.real
}

// -----

modelica.package @P {
    // expected-error @+1 {{is not allowed within packages}}
    modelica.equations {
    }
}

// -----

modelica.function @f {
    // expected-error @+1 {{is not allowed within functions}}
    modelica.equations {
    }
}

// -----

modelica.function @f {
    // expected-error @+1 {{functions can't have initial algorithms}}
    modelica.algorithm initial {
    }
}

// -----

modelica.operator @"'+'" {
    // expected-error @+1 {{is not allowed within operators}}
    modelica.model @M {
    }
}

// -----

modelica.package @P {
    // expected-error @+1 {{instantiated classes must be top-level definitions}}
    modelica.model @M {
        modelica.variable @x : !modelica.real
    }
}

// -----

// expected-error @+1 {{instantiated classes can contain only variables, instances, and equation and algorithm sections}}
modelica.model @M {
    modelica.component @x : @Real -> !modelica.real
    modelica.variable @y : !modelica.real
}
