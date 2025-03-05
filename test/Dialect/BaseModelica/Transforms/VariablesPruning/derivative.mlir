// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// CHECK-LABEL: @outputDerivedVar

bmodelica.model @outputDerivedVar der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, output>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, output>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real>

    // COM: der_x = 1
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @der_x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, indices = {}, match = @der_x
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @der_x
    // CHECK-NEXT:  }
}

// -----

// CHECK-LABEL: @outputDerivativeVar

bmodelica.model @outputDerivativeVar der = [<@x, @der_x>] {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-DAG: bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG: bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, output>

    // COM: der_x = 1
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @der_x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0, indices = {}, match = @der_x
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @der_x
    // CHECK-NEXT:  }
}
