// RUN: modelica-opt %s --split-input-file --inline-functions | FileCheck %s

// Used default value.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-DAG:      %[[a:.*]] = modelica.variable_get @a
// CHECK-DAG:      %[[default:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-DAG:      %[[add:.*]] = modelica.add %[[a]], %[[default]]
// CHECK-DAG:      %[[b:.*]] = modelica.variable_get @b
// CHECK-DAG:      %[[lhs:.*]] = modelica.equation_side %[[add]]
// CHECK-DAG:      %[[rhs:.*]] = modelica.equation_side %[[b]]
// CHECK-NEXT:     modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK:       }

modelica.function @Foo attributes {inline = true} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.default @y {
        %0 = modelica.constant #modelica.real<0.0>
        modelica.yield %0 : !modelica.real
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}

modelica.model @Test {
    modelica.variable @a : !modelica.variable<!modelica.real>
    modelica.variable @b : !modelica.variable<!modelica.real>

    modelica.equation {
        %0 = modelica.variable_get @a : !modelica.real
        %1 = modelica.call @Foo(%0) : (!modelica.real) -> !modelica.real
        %2 = modelica.variable_get @b : !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}

// -----

// Unused default value.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-DAG:      %[[a:.*]] = modelica.variable_get @a
// CHECK-DAG:      %[[add:.*]] = modelica.add %[[a]], %[[a]]
// CHECK-DAG:      %[[b:.*]] = modelica.variable_get @b
// CHECK-DAG:      %[[lhs:.*]] = modelica.equation_side %[[add]]
// CHECK-DAG:      %[[rhs:.*]] = modelica.equation_side %[[b]]
// CHECK-NEXT:     modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK:       }

modelica.function @Foo attributes {inline = true} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.default @y {
        %0 = modelica.constant #modelica.real<0.0>
        modelica.yield %0 : !modelica.real
    }

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}

modelica.model @Test {
    modelica.variable @a : !modelica.variable<!modelica.real>
    modelica.variable @b : !modelica.variable<!modelica.real>

    modelica.equation {
        %0 = modelica.variable_get @a : !modelica.real
        %1 = modelica.call @Foo(%0, %0) : (!modelica.real, !modelica.real) -> !modelica.real
        %2 = modelica.variable_get @b : !modelica.real
        %3 = modelica.equation_side %1 : tuple<!modelica.real>
        %4 = modelica.equation_side %2 : tuple<!modelica.real>
        modelica.equation_sides %3, %4 : tuple<!modelica.real>, tuple<!modelica.real>
    }
}
