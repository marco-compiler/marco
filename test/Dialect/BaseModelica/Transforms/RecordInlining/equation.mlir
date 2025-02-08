// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @componentsEquality

bmodelica.model @componentsEquality {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica<record @R>
            %1 = bmodelica.component_get %0, @x : !bmodelica<record @R> -> !bmodelica.real
            %2 = bmodelica.variable_get @r : !bmodelica<record @R>
            %3 = bmodelica.component_get %2, @y : !bmodelica<record @R> -> !bmodelica.real
            %4 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.real
            // CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.real
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[y]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}

// -----

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @componentsEqualitySharedRecordGet

bmodelica.model @componentsEqualitySharedRecordGet {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica<record @R>
            %1 = bmodelica.component_get %0, @x : !bmodelica<record @R> -> !bmodelica.real
            %2 = bmodelica.component_get %0, @y : !bmodelica<record @R> -> !bmodelica.real
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
            %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
            bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.real
            // CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.real
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[y]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}

// -----

// COM: Equality between two records.

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @recordsEquality

bmodelica.model @recordsEquality {
    bmodelica.variable @r1 : !bmodelica.variable<!bmodelica<record @R>>
    bmodelica.variable @r2 : !bmodelica.variable<!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r1 : !bmodelica<record @R>
            %1 = bmodelica.variable_get @r2 : !bmodelica<record @R>
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica<record @R>>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica<record @R>>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica<record @R>>, tuple<!bmodelica<record @R>>

            // CHECK-DAG:   %[[x_1:.*]] = bmodelica.variable_get @r1.x : !bmodelica.real
            // CHECK-DAG:   %[[y_1:.*]] = bmodelica.variable_get @r1.y : !bmodelica.real
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x_1]], %[[y_1]]
            // CHECK-DAG:   %[[x_2:.*]] = bmodelica.variable_get @r2.x : !bmodelica.real
            // CHECK-DAG:   %[[y_2:.*]] = bmodelica.variable_get @r2.y : !bmodelica.real
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[x_2]], %[[y_2]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}
