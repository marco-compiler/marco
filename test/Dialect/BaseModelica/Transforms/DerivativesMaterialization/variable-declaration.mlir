// RUN: modelica-opt %s --split-input-file --derivatives-materialization | FileCheck %s

// CHECK-LABEL: @scalarVariable
// CHECK-SAME: der = [<@x, @der_x>]

bmodelica.model @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
    }
}

// -----

// CHECK-LABEL: @arrayVariable1d
// CHECK-SAME: der = [<@x, @der_x, {[0,1]}>]

bmodelica.model @arrayVariable1d {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<2x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    %t1 = bmodelica.equation_template inductions = [] attributes {id = "eq1"} {
        %0 = bmodelica.variable_get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.tensor_extract %0[%1] : tensor<2x!bmodelica.real>
        %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.constant #bmodelica<real 0.0>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
        bmodelica.equation_instance %t1
    }
}

// -----

// CHECK-LABEL: @arrayVariable2d
// CHECK-SAME: der = [<@x, @der_x, {[3,5][12,14]}>]

bmodelica.model @arrayVariable2d {
    bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<10x20x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<10x20x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x20x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i0, %i1] : tensor<10x20x!bmodelica.real>
        %2 = bmodelica.der %1 : !bmodelica.real -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica<real 3.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[3,5][12,14]}
    }
}

// -----

// COM: Array variable with not all indices being derived.

// CHECK-LABEL: @partialArrayVariable
// CHECK-SAME: der = [<@x, @der_x, {[3,4]}>]

bmodelica.model @partialArrayVariable {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<10x!bmodelica.real>

    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.real>
        %1 = bmodelica.constant 3 : index
        %2 = bmodelica.constant 4 : index
        %3 = bmodelica.tensor_extract %0[%1] : tensor<10x!bmodelica.real>
        %4 = bmodelica.tensor_extract %0[%2] : tensor<10x!bmodelica.real>
        %5 = bmodelica.der %3 : !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.der %4 : !bmodelica.real -> !bmodelica.real
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.dynamic {
        bmodelica.equation_instance %t0
    }
}

// -----

// CHECK-LABEL: @algorithm
// CHECK-SAME: der = [<@x, @der_x, {[0,4]}>]

bmodelica.model @algorithm {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
            %1 = bmodelica.constant 3 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<5x!bmodelica.real>
            %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
            bmodelica.variable_set @x[%1], %3 : index, !bmodelica.real
        }
    }
}

// Check variable declaration and derivatives map.

// CHECK-LABEL: @initialAlgorithm
// CHECK-SAME: der = [<@x, @der_x, {[0,4]}>]

bmodelica.model @initialAlgorithm {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<5x!bmodelica.real>

    bmodelica.initial {
        bmodelica.algorithm {
            %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
            %1 = bmodelica.constant 3 : index
            %2 = bmodelica.tensor_extract %0[%1] : tensor<5x!bmodelica.real>
            %3 = bmodelica.der %2 : !bmodelica.real -> !bmodelica.real
            bmodelica.variable_set @x[%1], %3 : index, !bmodelica.real
        }
    }
}
