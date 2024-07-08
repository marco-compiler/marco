// RUN: modelica-opt %s --split-input-file --inline-records | FileCheck %s

// Scalar value in 1-D variable.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : tensor<2x!bmodelica.real>
// CHECK:           %[[x_insert:.*]] = bmodelica.tensor_insert %[[value]], %[[x]][%[[index]]]
// CHECK:           bmodelica.variable_set @r.x, %[[x_insert]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = bmodelica.constant #bmodelica<real 1.0>
        bmodelica.variable_component_set @r[%0]::@x, %1 : index, !bmodelica.real
    }
}

// -----

// Scalar value in 2-D variable.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[x_insert:.*]] = bmodelica.tensor_insert %[[value]], %[[x]][%[[index]], %[[index]]]
// CHECK:           bmodelica.variable_set @r.x, %[[x_insert]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x3x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = bmodelica.constant #bmodelica<real 1.0>
        bmodelica.variable_component_set @r[%0, %0]::@x, %1 : index, index, !bmodelica.real
    }
}

// -----

// Tensor value.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK:           %[[value:.*]] = tensor.empty()
// CHECK:           bmodelica.variable_set @r.x, %[[value]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = tensor.empty() : tensor<2x!bmodelica.real>
        bmodelica.variable_component_set @r::@x, %1 : tensor<2x!bmodelica.real>
    }
}

// -----

// Tensor value.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[unbounded:.*]] = bmodelica.unbounded_range
// CHECK-DAG:       %[[value:.*]] = tensor.empty()
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[x_insert:.*]] = bmodelica.tensor_insert_slice %[[value]], %[[x]][%[[index]], %[[unbounded]]]
// CHECK:           bmodelica.variable_set @r.x, %[[x_insert]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x3x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = tensor.empty() : tensor<3x!bmodelica.real>
        bmodelica.variable_component_set @r[%0]::@x, %1 : index, tensor<3x!bmodelica.real>
    }
}

// -----

// Tensor value.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[value:.*]] = tensor.empty()
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[x_insert:.*]] = bmodelica.tensor_insert_slice %[[value]], %[[x]][%[[index]]]
// CHECK:           bmodelica.variable_set @r.x, %[[x_insert]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = tensor.empty() : tensor<3x!bmodelica.real>
        bmodelica.variable_component_set @r[%0]::@x, %1 : index, tensor<3x!bmodelica.real>
    }
}

// -----

// Tensor value.

// CHECK-LABEL: @Test
// CHECK-DAG:   bmodelica.variable @r.x : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK-DAG:   bmodelica.variable @r.y : !bmodelica.variable<2x3x!bmodelica.real>
// CHECK:       bmodelica.algorithm {
// CHECK-DAG:       %[[unbounded:.*]] = bmodelica.unbounded_range
// CHECK-DAG:       %[[index:.*]] = bmodelica.constant 0 : index
// CHECK-DAG:       %[[value:.*]] = tensor.empty()
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @r.x : tensor<2x3x!bmodelica.real>
// CHECK:           %[[x_insert:.*]] = bmodelica.tensor_insert_slice %[[value]], %[[x]][%[[unbounded]], %[[index]]]
// CHECK:           bmodelica.variable_set @r.x, %[[x_insert]]
// CHECK-NEXT:  }

bmodelica.record @R {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<3x!bmodelica.real>
}

bmodelica.function @Test {
    bmodelica.variable @r : !bmodelica.variable<2x!bmodelica<record @R>>

    bmodelica.algorithm {
        %0 = bmodelica.constant 0 : index
        %1 = tensor.empty() : tensor<2x!bmodelica.real>
        bmodelica.variable_component_set @r::@x[%0], %1 : index, tensor<2x!bmodelica.real>
    }
}
