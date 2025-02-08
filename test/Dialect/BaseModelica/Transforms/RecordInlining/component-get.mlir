// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @extract

bmodelica.model @extract {
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : tensor<3x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<3x!bmodelica<record @R>>
            %3 = bmodelica.component_get %2, @x : !bmodelica<record @R> -> !bmodelica.real
            %4 = bmodelica.component_get %2, @y : !bmodelica<record @R> -> !bmodelica.real
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[index:.*]] = bmodelica.constant 0 : index
            // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @r.x : tensor<3x!bmodelica.real>
            // CHECK-DAG:   %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[index]]]
            // CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @r.y : tensor<3x!bmodelica.real>
            // CHECK-DAG:   %[[y_extract:.*]] = bmodelica.tensor_extract %[[y]][%[[index]]]
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[y_extract]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}

// -----

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

// CHECK-LABEL: @viewAndExtract

bmodelica.model @viewAndExtract {
    bmodelica.variable @r : !bmodelica.variable<3x5x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : tensor<3x5x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.tensor_view %0[%1] : tensor<3x5x!bmodelica<record @R>>, index -> tensor<5x!bmodelica<record @R>>
            %3 = bmodelica.tensor_extract %2[%1] : tensor<5x!bmodelica<record @R>>
            %4 = bmodelica.component_get %3, @x : !bmodelica<record @R> -> !bmodelica.real
            %5 = bmodelica.component_get %3, @y : !bmodelica<record @R> -> !bmodelica.real
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
            bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[index:.*]] = bmodelica.constant 0 : index
            // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @r.x : tensor<3x5x!bmodelica.real>
            // CHECK-DAG:   %[[x_view:.*]] = bmodelica.tensor_view %[[x]][%[[index]]]
            // CHECK-DAG:   %[[x_extract:.*]] = bmodelica.tensor_extract %[[x_view]][%[[index]]]
            // CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @r.y : tensor<3x5x!bmodelica.real>
            // CHECK-DAG:   %[[y_view:.*]] = bmodelica.tensor_view %[[y]][%[[index]]]
            // CHECK-DAG:   %[[y_extract:.*]] = bmodelica.tensor_extract %[[y_view]][%[[index]]]
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[y_extract]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}

// -----

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @r : !bmodelica.variable<!bmodelica<record @R>, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
}

// CHECK-LABEL: @callArgument

bmodelica.model @callArgument {
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @r : tensor<3x!bmodelica<record @R>>
            %1 = bmodelica.constant 0 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<3x!bmodelica<record @R>>
            %3 = bmodelica.call @Foo(%2) : (!bmodelica<record @R>) -> !bmodelica.real
            %4 = bmodelica.constant #bmodelica<real 0.0>
            %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[index:.*]] = bmodelica.constant 0 : index
            // CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @r.x : tensor<3x!bmodelica.real>
            // CHECK-DAG:   %[[x_extract:.*]] = bmodelica.tensor_extract %[[x]][%[[index]]]
            // CHECK-DAG:   %[[y:.*]] = bmodelica.variable_get @r.y : tensor<3x!bmodelica.real>
            // CHECK-DAG:   %[[y_extract:.*]] = bmodelica.tensor_extract %[[y]][%[[index]]]
            // CHECK:       %[[call:.*]] = bmodelica.call @Foo(%[[x_extract]], %[[y_extract]]) : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            // CHECK:       %[[lhs:.*]] = bmodelica.equation_side %[[call]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %{{.*}}
        }
    }
}

// -----

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Foo {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @r : !bmodelica.variable<3x!bmodelica<record @R>, output>
}

// CHECK-LABEL: @callResult

bmodelica.model @callResult {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.real
            %1 = bmodelica.variable_get @y : !bmodelica.real
            %2 = bmodelica.call @Foo(%0, %1) : (!bmodelica.real, !bmodelica.real) -> tensor<3x!bmodelica<record @R>>
            %3 = bmodelica.constant 0 : index
            %4 = bmodelica.tensor_extract %2[%3] : tensor<3x!bmodelica<record @R>>
            %5 = bmodelica.component_get %4, @x : !bmodelica<record @R> -> !bmodelica.real
            %6 = bmodelica.component_get %4, @y : !bmodelica<record @R> -> !bmodelica.real
            %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
            %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
            bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>

            // CHECK-DAG:   %[[index:.*]] = bmodelica.constant 0 : index
            // CHECK-DAG:   %[[call:.*]]:2 = bmodelica.call @Foo(%{{.*}}, %{{.*}}) : (!bmodelica.real, !bmodelica.real) -> (tensor<3x!bmodelica.real>, tensor<3x!bmodelica.real>)
            // CHECK-DAG:   %[[x_extract:.*]] = bmodelica.tensor_extract %[[call]]#0[%[[index]]]
            // CHECK-DAG:   %[[y_extract:.*]] = bmodelica.tensor_extract %[[call]]#1[%[[index]]]
            // CHECK-DAG:   %[[lhs:.*]] = bmodelica.equation_side %[[x_extract]]
            // CHECK-DAG:   %[[rhs:.*]] = bmodelica.equation_side %[[y_extract]]
            // CHECK:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
        }
    }
}
