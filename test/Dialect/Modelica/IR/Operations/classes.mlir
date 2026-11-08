// RUN: marco-opt %s --split-input-file | FileCheck %s

// Class definitions.

// CHECK:       modelica.class connector @Pin {
// CHECK-NEXT:      modelica.component @v : @Real -> !modelica.real
// CHECK-NEXT:      modelica.component flow @i : @Real -> !modelica.real
// CHECK-NEXT:  }
// CHECK:       modelica.class partial model @TwoPin {
// CHECK:       modelica.class encapsulated package @P {
// CHECK:       modelica.class expandable_connector @Bus {
// CHECK:       modelica.class operator_record @Complex {
// CHECK:       modelica.class impure function @f {
// CHECK:       modelica.class final operator_function @g {

modelica.class connector @Pin {
    modelica.component @v : @Real -> !modelica.real
    modelica.component flow @i : @Real -> !modelica.real
}

modelica.class partial model @TwoPin {
}

modelica.class encapsulated package @P {
}

modelica.class expandable_connector @Bus {
}

modelica.class operator_record @Complex {
}

modelica.class impure function @f {
}

modelica.class final operator_function @g {
}

// -----

// Short class definitions.

// CHECK:       modelica.class short type @Voltage {
// CHECK-NEXT:      modelica.extends @Real {
// CHECK-NEXT:          modelica.modify @unit {
// CHECK:       modelica.class short type @Vector3 {
// CHECK-NEXT:      modelica.extends @Real dims [3]
// CHECK:       modelica.class short connector @RealInput {
// CHECK-NEXT:      modelica.extends @Real input
// CHECK:       modelica.class short type @Matrix {
// CHECK-NEXT:      modelica.extends @Real dims [?, :] {
// CHECK-NEXT:          %[[n:.*]] = modelica.ref @n : !modelica.int
// CHECK-NEXT:          modelica.yield %[[n]] : !modelica.int
// CHECK-NEXT:      }

modelica.class short type @Voltage {
    modelica.extends @Real {
        modelica.modify @unit {
            %0 = modelica.constant #modelica<string "V"> : !modelica.string
            modelica.bind %0 : !modelica.string
        }
    }
}

modelica.class short type @Vector3 {
    modelica.extends @Real dims [3]
}

modelica.class short connector @RealInput {
    modelica.extends @Real input
}

modelica.class short type @Matrix {
    modelica.extends @Real dims [?, :] {
        %0 = modelica.ref @n : !modelica.int
        modelica.yield %0 : !modelica.int
    }
}

// -----

// Class-extends redeclarations and derivative
// definitions.

// CHECK:       modelica.class redeclare replaceable package extends @Medium {
// CHECK-NEXT:      modelica.extends inherited {
// CHECK-NEXT:          modelica.modify @nX {
// CHECK:       modelica.class function @df der(@f, "x", "y") {

modelica.class package @Library {
    modelica.class redeclare replaceable package extends @Medium {
        modelica.extends inherited {
            modelica.modify @nX {
                %0 = modelica.constant #modelica<int 2> : !modelica.int
                modelica.bind %0 : !modelica.int
            }
        }
    }

    modelica.class function @df der(@f, "x", "y") {
    }
}

// -----

// Enumerations.

// CHECK:       modelica.enumeration @Colors {
// CHECK-NEXT:      modelica.enum_literal @red
// CHECK-NEXT:      modelica.enum_literal @green "The green color"
// CHECK-NEXT:  }
// CHECK:       modelica.enumeration unspecified @Free {
// CHECK:       modelica.enumeration replaceable @Alternatives {
// CHECK-NEXT:      modelica.enum_literal @a
// CHECK-NEXT:      modelica.constrainedby @Base

modelica.class package @Types {
    modelica.enumeration @Colors {
        modelica.enum_literal @red
        modelica.enum_literal @green "The green color"
    }

    modelica.enumeration unspecified @Free {
    }

    modelica.enumeration replaceable @Alternatives {
        modelica.enum_literal @a
        modelica.constrainedby @Base
    }
}

// -----

// Components.

// CHECK-LABEL: modelica.class model @Components
// CHECK: modelica.component protected parameter @a : @Real -> !modelica.real
// CHECK: modelica.component final constant input @b : @Integer -> !modelica.int
// CHECK: modelica.component inner outer @world : @World -> !modelica.object<@World>
// CHECK: modelica.component stream discrete output @h : @Real -> !modelica.real
// CHECK: modelica.component @x : @Real -> tensor<2x?x3x2x!modelica.real> dims [Boolean, :, @Colors, ?] {
// CHECK: modelica.component @c : @Resistor -> !modelica.object<@Resistor> if {
// CHECK: modelica.component replaceable @medium : @Water -> !modelica.object<@PartialMedium> constrainedby @PartialMedium {
// CHECK: modelica.component @d : @Real -> !modelica.real attributes {annotation = {Evaluate = true}, description = "A component"}

modelica.class model @Components {
    modelica.component protected parameter @a : @Real -> !modelica.real
    modelica.component final constant input @b : @Integer -> !modelica.int
    modelica.component inner outer @world : @World -> !modelica.object<@World>
    modelica.component stream discrete output @h : @Real -> !modelica.real

    // Real[Colors, 2] x[Boolean, :] = ...: the declaration dimensions precede
    // the ones of the type specifier.
    modelica.component @x : @Real -> tensor<2x?x3x2x!modelica.real> dims [Boolean, :, @Colors, ?] {
        %0 = modelica.constant #modelica<int 2> : !modelica.int
        modelica.yield %0 : !modelica.int
    }

    modelica.component @c : @Resistor -> !modelica.object<@Resistor> if {
        %0 = modelica.ref @useResistor : !modelica.bool
        modelica.yield %0 : !modelica.bool
    }

    modelica.component replaceable @medium : @Water -> !modelica.object<@PartialMedium> constrainedby @PartialMedium {
        modelica.modify @p {
            %0 = modelica.constant #modelica<real 1.0e5> : !modelica.real
            modelica.bind %0 : !modelica.real
        }
    }

    modelica.component @d : @Real -> !modelica.real attributes {description = "A component", annotation = {Evaluate = true}}
}

// -----

// Modifications, including composite names, 'each',
// 'final', 'break' and redeclarations.

// CHECK-LABEL: modelica.class model @Modifications
// CHECK:       modelica.component @r : @Resistor -> tensor<3x!modelica.object<@Resistor>> {
// CHECK-NEXT:      modelica.modify each final @R {
// CHECK:           modelica.modify @heatPort::@T {
// CHECK:           modelica.modify @v {
// CHECK-NEXT:          modelica.bind break
// CHECK-NEXT:      }
// CHECK-NEXT:  } dims [3]
// CHECK:       modelica.component @s : @System -> !modelica.object<@System> {
// CHECK-NEXT:      modelica.component redeclare @medium : @Water -> !modelica.object<@PartialMedium>
// CHECK-NEXT:      modelica.class redeclare short model @Controller {
// CHECK-NEXT:          modelica.extends @PI
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.modify @sub {
// CHECK-NEXT:          modelica.component redeclare each @x : @Real -> !modelica.real
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.class model @Modifications {
    modelica.component @r : @Resistor -> tensor<3x!modelica.object<@Resistor>> {
        modelica.modify each final @R {
            %0 = modelica.constant #modelica<real 2.0> : !modelica.real
            modelica.bind %0 : !modelica.real
        }
        modelica.modify @heatPort::@T {
            %0 = modelica.constant #modelica<real 300.0> : !modelica.real
            modelica.bind %0 : !modelica.real
        }
        modelica.modify @v {
            modelica.bind break
        }
    } dims [3]

    modelica.component @s : @System -> !modelica.object<@System> {
        modelica.component redeclare @medium : @Water -> !modelica.object<@PartialMedium>
        modelica.class redeclare short model @Controller {
            modelica.extends @PI
        }
        modelica.modify @sub {
            modelica.component redeclare each @x : @Real -> !modelica.real
        }
    }
}

// -----

// Extends-clauses and selective model extension.

// CHECK-LABEL: modelica.class model @Derived
// CHECK:       modelica.extends @Base
// CHECK:       modelica.extends protected @Other attributes {annotation = {IconMap = {primitivesVisible = false}}}
// CHECK:       modelica.extends @System {
// CHECK-NEXT:      modelica.deselect @friction
// CHECK-NEXT:      modelica.deselect_connect {
// CHECK:               modelica.yield
// CHECK-NEXT:      }
// CHECK-NEXT:  }

modelica.class model @Derived {
    modelica.extends @Base
    modelica.extends protected @Other attributes {annotation = {IconMap = {primitivesVisible = false}}}
    modelica.extends @System {
        modelica.deselect @friction
        modelica.deselect_connect {
            %0 = modelica.ref @a : !modelica.object<@Pin>
            %1 = modelica.ref @b : !modelica.object<@Pin>
            modelica.yield %0, %1 : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
    }
}

// -----

// Import-clauses and global names.

// CHECK-LABEL: modelica.class package @Imports
// CHECK: modelica.import @Modelica::@Units::@SI
// CHECK: modelica.import @SI = @Modelica::@Units::@SI
// CHECK: modelica.import protected @Modelica::@Constants unqualified
// CHECK: modelica.import @Modelica::@Math {@sin, @cos}
// CHECK: modelica.component @x : @"."::@Modelica::@Units::@SI::@Voltage -> !modelica.real

modelica.class package @Imports {
    modelica.import @Modelica::@Units::@SI
    modelica.import @SI = @Modelica::@Units::@SI
    modelica.import protected @Modelica::@Constants unqualified
    modelica.import @Modelica::@Math {@sin, @cos}

    modelica.class model @M {
        modelica.component @x : @"."::@Modelica::@Units::@SI::@Voltage -> !modelica.real
    }
}

// -----

// Sections and external functions.

// CHECK-LABEL: modelica.class function @externalFunction
// CHECK:       modelica.external "C" "f" output {
// CHECK:           modelica.yield
// CHECK-NEXT:  } {annotation = {Library = "f"}}
// CHECK-LABEL: modelica.class function @defaultExternal
// CHECK:       modelica.external "FORTRAN 77"
// CHECK-LABEL: modelica.class model @Sections
// CHECK: modelica.equations {
// CHECK: modelica.equations initial {
// CHECK: modelica.algorithm {
// CHECK: modelica.algorithm initial {

modelica.class function @externalFunction {
    modelica.component input @x : @Real -> tensor<?x!modelica.real> dims [:]
    modelica.component output @y : @Real -> !modelica.real
    modelica.external "C" "f" output {
        %y = modelica.ref @y : !modelica.real
        %x = modelica.ref @x : tensor<?x!modelica.real>
        %zero = modelica.constant #modelica<int 0> : !modelica.int
        %n = modelica.size %x, %zero : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
        modelica.yield %y, %x, %n : !modelica.real, tensor<?x!modelica.real>, !modelica.int
    } {annotation = {Library = "f"}}
}

modelica.class function @defaultExternal {
    modelica.external "FORTRAN 77"
}

modelica.class model @Sections {
    modelica.equations {
    }
    modelica.equations initial {
    }
    modelica.algorithm {
    }
    modelica.algorithm initial {
    }
}
