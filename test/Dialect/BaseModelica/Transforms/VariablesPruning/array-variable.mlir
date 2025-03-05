// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// CHECK-LABEL: @arrayDependency

bmodelica.model @arrayDependency {
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-DAG:  bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    // COM: x[i] = 0
    %t0 = bmodelica.equation_template inductions = [%i] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %1 = bmodelica.tensor_extract %0[%i] : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y = x[0]
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : tensor<5x!bmodelica.real>
        %2 = bmodelica.constant 0 : index
        %3 = bmodelica.tensor_extract %1[%2] : tensor<5x!bmodelica.real>
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, indices = {[0,4]}, match = <@x, {[0,4]}>
        bmodelica.matched_equation_instance %t1, indices = {}, match = @y
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t0]], indices = {[0,4]}, match = <@x, {[0,4]}>
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @y
    // CHECK-NEXT:  }
}
