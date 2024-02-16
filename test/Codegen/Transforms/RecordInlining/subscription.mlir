// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// Get through load.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[index:.*]] = modelica.constant 0 : index
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<3x!modelica.real>
// CHECK-DAG:       %[[subscription_x:.*]] = modelica.subscription %[[x]][%[[index]]]
// CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[subscription_x]]
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<3x!modelica.real>
// CHECK-DAG:       %[[subscription_y:.*]] = modelica.subscription %[[y]][%[[index]]]
// CHECK-DAG:       %[[load_y:.*]] = modelica.load %[[subscription_y]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[load_y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<3x!modelica<record @R>>

    modelica.main_model {
        modelica.equation {
            %0 = modelica.variable_get @r : !modelica.array<3x!modelica<record @R>>
            %1 = modelica.constant 0 : index
            %2 = modelica.load %0[%1] : !modelica.array<3x!modelica<record @R>>
            %3 = modelica.component_get %2, @x : !modelica<record @R> -> !modelica.real
            %4 = modelica.component_get %2, @y : !modelica<record @R> -> !modelica.real
            %5 = modelica.equation_side %3 : tuple<!modelica.real>
            %6 = modelica.equation_side %4 : tuple<!modelica.real>
            modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}

// -----

// Get through load and subscription.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x5x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x5x!modelica.real>
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[index:.*]] = modelica.constant 0 : index
// CHECK-DAG:       %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<3x5x!modelica.real>
// CHECK-DAG:       %[[subscription_x_1:.*]] = modelica.subscription %[[x]][%[[index]]]
// CHECK-DAG:       %[[subscription_x_2:.*]] = modelica.subscription %[[subscription_x_1]][%[[index]]]
// CHECK-DAG:       %[[load_x:.*]] = modelica.load %[[subscription_x_2]]
// CHECK-DAG:       %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<3x5x!modelica.real>
// CHECK-DAG:       %[[subscription_y_1:.*]] = modelica.subscription %[[y]][%[[index]]]
// CHECK-DAG:       %[[subscription_y_2:.*]] = modelica.subscription %[[subscription_y_1]][%[[index]]]
// CHECK-DAG:       %[[load_y:.*]] = modelica.load %[[subscription_y_2]]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[load_y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<3x5x!modelica<record @R>>

    modelica.main_model {
        modelica.equation {
            %0 = modelica.variable_get @r : !modelica.array<3x5x!modelica<record @R>>
            %1 = modelica.constant 0 : index
            %2 = modelica.subscription %0[%1] : !modelica.array<3x5x!modelica<record @R>>, index -> !modelica.array<5x!modelica<record @R>>
            %3 = modelica.load %2[%1] : !modelica.array<5x!modelica<record @R>>
            %4 = modelica.component_get %3, @x : !modelica<record @R> -> !modelica.real
            %5 = modelica.component_get %3, @y : !modelica<record @R> -> !modelica.real
            %6 = modelica.equation_side %4 : tuple<!modelica.real>
            %7 = modelica.equation_side %5 : tuple<!modelica.real>
            modelica.equation_sides %6, %7 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}

// -----

// Set through load.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x!modelica.real>
// CHECK:       modelica.algorithm {
// CHECK:           %[[value:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:           %[[index:.*]] = modelica.constant 0 : index
// CHECK:           %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<3x!modelica.real>
// CHECK:           %[[subscription_x:.*]] = modelica.subscription %[[x]][%[[index]]]
// CHECK:           modelica.assignment %[[subscription_x]], %[[value]]
// CHECK:           %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<3x!modelica.real>
// CHECK:           %[[subscription_y:.*]] = modelica.subscription %[[y]][%[[index]]]
// CHECK:           modelica.assignment %[[subscription_y]], %[[value]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<3x!modelica<record @R>>

    modelica.algorithm {
        %0 = modelica.variable_get @r : !modelica.array<3x!modelica<record @R>>
        %1 = modelica.constant 0 : index
        %2 = modelica.load %0[%1] : !modelica.array<3x!modelica<record @R>>
        %3 = modelica.constant #modelica.real<1.0>
        %4 = modelica.constant #modelica.real<1.0>
        modelica.component_set %2, @x, %3 : !modelica<record @R>, !modelica.real
        modelica.component_set %2, @y, %4 : !modelica<record @R>, !modelica.real
    }
}

// -----

// Set through subscription.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x5x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x5x!modelica.real>
// CHECK:       modelica.algorithm {
// CHECK:           %[[value:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:           %[[index:.*]] = modelica.constant 0 : index
// CHECK:           %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<3x5x!modelica.real>
// CHECK:           %[[subscription_x_1:.*]] = modelica.subscription %[[x]][%[[index]]]
// CHECK:           %[[subscription_x_2:.*]] = modelica.subscription %[[subscription_x_1]][%[[index]]]
// CHECK:           modelica.assignment %[[subscription_x_2]], %[[value]]
// CHECK:           %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<3x5x!modelica.real>
// CHECK:           %[[subscription_y_1:.*]] = modelica.subscription %[[y]][%[[index]]]
// CHECK:           %[[subscription_y_2:.*]] = modelica.subscription %[[subscription_y_1]][%[[index]]]
// CHECK:           modelica.assignment %[[subscription_y_2]], %[[value]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Test {
    modelica.variable @r : !modelica.variable<3x5x!modelica<record @R>>

    modelica.algorithm {
        %0 = modelica.variable_get @r : !modelica.array<3x5x!modelica<record @R>>
        %1 = modelica.constant 0 : index
        %2 = modelica.subscription %0[%1] : !modelica.array<3x5x!modelica<record @R>>, index -> !modelica.array<5x!modelica<record @R>>
        %3 = modelica.load %2[%1] : !modelica.array<5x!modelica<record @R>>
        %4 = modelica.constant #modelica.real<1.0>
        %5 = modelica.constant #modelica.real<1.0>
        modelica.component_set %3, @x, %4 : !modelica<record @R>, !modelica.real
        modelica.component_set %3, @y, %5 : !modelica<record @R>, !modelica.real
    }
}

// -----

// Call argument.

// CHECK-LABEL: @Test
// CHECK: modelica.variable @r.x : !modelica.variable<3x!modelica.real>
// CHECK: modelica.variable @r.y : !modelica.variable<3x!modelica.real>
// CHECK:       modelica.equation {
// CHECK:           %[[index:.*]] = modelica.constant 0 : index
// CHECK:           %[[x:.*]] = modelica.variable_get @r.x : !modelica.array<3x!modelica.real>
// CHECK:           %[[subscription_x:.*]] = modelica.subscription %[[x]][%[[index]]]
// CHECK:           %[[load_x:.*]] = modelica.load %[[subscription_x]][]
// CHECK:           %[[y:.*]] = modelica.variable_get @r.y : !modelica.array<3x!modelica.real>
// CHECK:           %[[subscription_y:.*]] = modelica.subscription %[[y]][%[[index]]]
// CHECK:           %[[load_y:.*]] = modelica.load %[[subscription_y]][]
// CHECK:           %[[call:.*]] = modelica.call @Foo(%[[load_x]], %[[load_y]]) : (!modelica.real, !modelica.real) -> !modelica.real
// CHECK:           %[[lhs:.*]] = modelica.equation_side %[[call]]
// CHECK:           modelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Foo {
    modelica.variable @r : !modelica.variable<!modelica<record @R>, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>
}

modelica.model @Test {
    modelica.variable @r : !modelica.variable<3x!modelica<record @R>>

    modelica.main_model {
        modelica.equation {
            %0 = modelica.variable_get @r : !modelica.array<3x!modelica<record @R>>
            %1 = modelica.constant 0 : index
            %2 = modelica.load %0[%1] : !modelica.array<3x!modelica<record @R>>
            %3 = modelica.call @Foo(%2) : (!modelica<record @R>) -> !modelica.real
            %4 = modelica.constant #modelica.real<0.0>
            %5 = modelica.equation_side %3 : tuple<!modelica.real>
            %6 = modelica.equation_side %4 : tuple<!modelica.real>
            modelica.equation_sides %5, %6 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}

// -----

// Call result.

// CHECK-LABEL: @Test
// CHECK:       modelica.equation {
// CHECK-DAG:       %[[index:.*]] = modelica.constant 0 : index
// CHECK-DAG:       %[[call:.*]]:2 = modelica.call @Foo(%{{.*}}, %{{.*}}) : (!modelica.real, !modelica.real) -> (!modelica.array<3x!modelica.real>, !modelica.array<3x!modelica.real>)
// CHECK:           %[[subscription_x:.*]] = modelica.subscription %[[call]]#0[%[[index]]]
// CHECK:           %[[load_x:.*]] = modelica.load %[[subscription_x]][]
// CHECK:           %[[subscription_y:.*]] = modelica.subscription %[[call]]#1[%[[index]]]
// CHECK:           %[[load_y:.*]] = modelica.load %[[subscription_y]][]
// CHECK-DAG:       %[[lhs:.*]] = modelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = modelica.equation_side %[[load_y]]
// CHECK:           modelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

modelica.record @R {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>
}

modelica.function @Foo {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @r : !modelica.variable<3x!modelica<record @R>, output>
}

modelica.model @Test {
    modelica.variable @x : !modelica.variable<!modelica.real>
    modelica.variable @y : !modelica.variable<!modelica.real>

    modelica.main_model {
        modelica.equation {
            %0 = modelica.variable_get @x : !modelica.real
            %1 = modelica.variable_get @y : !modelica.real
            %2 = modelica.call @Foo(%0, %1) : (!modelica.real, !modelica.real) -> !modelica.array<3x!modelica<record @R>>
            %3 = modelica.constant 0 : index
            %4 = modelica.load %2[%3] : !modelica.array<3x!modelica<record @R>>
            %5 = modelica.component_get %4, @x : !modelica<record @R> -> !modelica.real
            %6 = modelica.component_get %4, @y : !modelica<record @R> -> !modelica.real
            %7 = modelica.equation_side %5 : tuple<!modelica.real>
            %8 = modelica.equation_side %6 : tuple<!modelica.real>
            modelica.equation_sides %7, %8 : tuple<!modelica.real>, tuple<!modelica.real>
        }
    }
}
