// RUN: modelica-opt %s --split-input-file --pure-function-inlining | FileCheck %s

bmodelica.function @Foo attributes {inline = true} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.default @y {
        %0 = bmodelica.constant #bmodelica<real 0.0>
        bmodelica.yield %0 : !bmodelica.real
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// CHECK-LABEL: @usedDefaultValue

bmodelica.model @usedDefaultValue {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @a : !bmodelica.real
            %1 = bmodelica.call @Foo(%0) : (!bmodelica.real) -> !bmodelica.real
            %2 = bmodelica.variable_get @b : !bmodelica.real
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[a:.*]] = bmodelica.variable_get @a
            // CHECK-DAG:   %[[default:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
            // CHECK-DAG:   %[[add:.*]] = bmodelica.add %[[a]], %[[default]]
            // CHECK-DAG:   %[[b:.*]] = bmodelica.variable_get @b
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[add]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[b]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}

// -----

bmodelica.function @Foo attributes {inline = true} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.default @y {
        %0 = bmodelica.constant #bmodelica<real 0.0>
        bmodelica.yield %0 : !bmodelica.real
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// CHECK-LABEL: @unusedDefaultValue

bmodelica.model @unusedDefaultValue {
    bmodelica.variable @a : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @a : !bmodelica.real
            %1 = bmodelica.call @Foo(%0, %0) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            %2 = bmodelica.variable_get @b : !bmodelica.real
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[a:.*]] = bmodelica.variable_get @a
            // CHECK-DAG:   %[[add:.*]] = bmodelica.add %[[a]], %[[a]]
            // CHECK-DAG:   %[[b:.*]] = bmodelica.variable_get @b
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[add]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[b]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}
