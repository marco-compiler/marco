// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// Get through load.

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<3x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<3x!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<3x!bmodelica.real>
// CHECK-DAG:       %[[subscription_x:.*]] = bmodelica.subscription %[[x]][%[[index]]]
// CHECK-DAG:       %[[load_x:.*]] = bmodelica.load %[[subscription_x]]
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<3x!bmodelica.real>
// CHECK-DAG:       %[[subscription_y:.*]] = bmodelica.subscription %[[y]][%[[index]]]
// CHECK-DAG:       %[[load_y:.*]] = bmodelica.load %[[subscription_y]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[load_y]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.model @Test {
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica.array<3x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.load %0[%1] : !bmodelica.array<3x!bmodelica<record @R>>
            %3 = bmodelica.component_get %2, @x : !bmodelica<record @R> -> !bmodelica.real
            %4 = bmodelica.component_get %2, @y : !bmodelica<record @R> -> !bmodelica.real
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}

// -----

// Get through load and subscription.

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<3x5x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<3x5x!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<3x5x!bmodelica.real>
// CHECK-DAG:       %[[subscription_x_1:.*]] = bmodelica.subscription %[[x]][%[[index]]]
// CHECK-DAG:       %[[subscription_x_2:.*]] = bmodelica.subscription %[[subscription_x_1]][%[[index]]]
// CHECK-DAG:       %[[load_x:.*]] = bmodelica.load %[[subscription_x_2]]
// CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<3x5x!bmodelica.real>
// CHECK-DAG:       %[[subscription_y_1:.*]] = bmodelica.subscription %[[y]][%[[index]]]
// CHECK-DAG:       %[[subscription_y_2:.*]] = bmodelica.subscription %[[subscription_y_1]][%[[index]]]
// CHECK-DAG:       %[[load_y:.*]] = bmodelica.load %[[subscription_y_2]]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[load_y]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.model @Test {
    bmodelica.variable @r : !bmodelica.variable<3x5x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica.array<3x5x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.subscription %0[%1] : !bmodelica.array<3x5x!bmodelica<record @R>>, index -> !bmodelica.array<5x!bmodelica<record @R>>
            %3 = bmodelica.load %2[%1] : !bmodelica.array<5x!bmodelica<record @R>>
            %4 = bmodelica.component_get %3, @x : !bmodelica<record @R> -> !bmodelica.real
            %5 = bmodelica.component_get %3, @y : !bmodelica<record @R> -> !bmodelica.real
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
            bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}

// -----

// Set through load.

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<3x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<3x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK:           %[[value:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:           %[[index:.*]] = bmodelica.constant 0 : index
// CHECK:           %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<3x!bmodelica.real>
// CHECK:           %[[subscription_x:.*]] = bmodelica.subscription %[[x]][%[[index]]]
// CHECK:           bmodelica.assignment %[[subscription_x]], %[[value]]
// CHECK:           %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<3x!bmodelica.real>
// CHECK:           %[[subscription_y:.*]] = bmodelica.subscription %[[y]][%[[index]]]
// CHECK:           bmodelica.assignment %[[subscription_y]], %[[value]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @r : !bmodelica.array<3x!bmodelica<record @R>>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.load %0[%1] : !bmodelica.array<3x!bmodelica<record @R>>
        %3 = bmodelica.constant #bmodelica.real<1.0>
        %4 = bmodelica.constant #bmodelica.real<1.0>
        bmodelica.component_set %2, @x, %3 : !bmodelica<record @R>, !bmodelica.real
        bmodelica.component_set %2, @y, %4 : !bmodelica<record @R>, !bmodelica.real
    }
}

// -----

// Set through subscription.

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<3x5x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<3x5x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK:           %[[value:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:           %[[index:.*]] = bmodelica.constant 0 : index
// CHECK:           %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<3x5x!bmodelica.real>
// CHECK:           %[[subscription_x_1:.*]] = bmodelica.subscription %[[x]][%[[index]]]
// CHECK:           %[[subscription_x_2:.*]] = bmodelica.subscription %[[subscription_x_1]][%[[index]]]
// CHECK:           bmodelica.assignment %[[subscription_x_2]], %[[value]]
// CHECK:           %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<3x5x!bmodelica.real>
// CHECK:           %[[subscription_y_1:.*]] = bmodelica.subscription %[[y]][%[[index]]]
// CHECK:           %[[subscription_y_2:.*]] = bmodelica.subscription %[[subscription_y_1]][%[[index]]]
// CHECK:           bmodelica.assignment %[[subscription_y_2]], %[[value]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<3x5x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @r : !bmodelica.array<3x5x!bmodelica<record @R>>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.subscription %0[%1] : !bmodelica.array<3x5x!bmodelica<record @R>>, index -> !bmodelica.array<5x!bmodelica<record @R>>
        %3 = bmodelica.load %2[%1] : !bmodelica.array<5x!bmodelica<record @R>>
        %4 = bmodelica.constant #bmodelica.real<1.0>
        %5 = bmodelica.constant #bmodelica.real<1.0>
        bmodelica.component_set %3, @x, %4 : !bmodelica<record @R>, !bmodelica.real
        bmodelica.component_set %3, @y, %5 : !bmodelica<record @R>, !bmodelica.real
    }
}

// -----

// Call argument.

// CHECK-LABEL: @Test
// CHECK: bmodelica.variable @r.x : !bmodelica.variable<3x!bmodelica.real>
// CHECK: bmodelica.variable @r.y : !bmodelica.variable<3x!bmodelica.real>
// CHECK:       bmodelica.equation {
// CHECK:           %[[index:.*]] = bmodelica.constant 0 : index
// CHECK:           %[[x:.*]] = bmodelica.variable_get @r.x : !bmodelica.array<3x!bmodelica.real>
// CHECK:           %[[subscription_x:.*]] = bmodelica.subscription %[[x]][%[[index]]]
// CHECK:           %[[load_x:.*]] = bmodelica.load %[[subscription_x]][]
// CHECK:           %[[y:.*]] = bmodelica.variable_get @r.y : !bmodelica.array<3x!bmodelica.real>
// CHECK:           %[[subscription_y:.*]] = bmodelica.subscription %[[y]][%[[index]]]
// CHECK:           %[[load_y:.*]] = bmodelica.load %[[subscription_y]][]
// CHECK:           %[[call:.*]] = bmodelica.call @Foo(%[[load_x]], %[[load_y]]) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK:           %[[lhs:.*]] = bmodelica.equation_side %[[call]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %{{.*}}
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
}

bmodelica.model @Test {
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : !bmodelica.array<3x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.load %0[%1] : !bmodelica.array<3x!bmodelica<record @R>>
            %3 = bmodelica.call @Foo(%2) : (!bmodelica<record @R>) -> !bmodelica.real
            %4 = bmodelica.constant #bmodelica.real<0.0>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}

// -----

// Call result.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.equation {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[call:.*]]:2 = bmodelica.call @Foo(%{{.*}}, %{{.*}}) : (!bmodelica.real, !bmodelica.real) -> (!bmodelica.array<3x!bmodelica.real>, !bmodelica.array<3x!bmodelica.real>)
// CHECK:           %[[subscription_x:.*]] = bmodelica.subscription %[[call]]#0[%[[index]]]
// CHECK:           %[[load_x:.*]] = bmodelica.load %[[subscription_x]][]
// CHECK:           %[[subscription_y:.*]] = bmodelica.subscription %[[call]]#1[%[[index]]]
// CHECK:           %[[load_y:.*]] = bmodelica.load %[[subscription_y]][]
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[load_x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[load_y]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>, output>
}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.variable_get @y : !bmodelica.real
            %2 = bmodelica.call @Foo(%0, %1) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.array<3x!bmodelica<record @R>>
            %3 = bmodelica.constant 0 : index
            %4 = bmodelica.load %2[%3] : !bmodelica.array<3x!bmodelica<record @R>>
            %5 = bmodelica.component_get %4, @x : !bmodelica<record @R> -> !bmodelica.real
            %6 = bmodelica.component_get %4, @y : !bmodelica<record @R> -> !bmodelica.real
            %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
            %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
            bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }
    }
}
