// RUN: modelica-opt %s --split-input-file --variables-pruning | FileCheck %s

// CHECK-LABEL: @InitialWithDependency

bmodelica.model @InitialWithDependency {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-DAG:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // COM: x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: z = y
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    // COM: z = 0
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0, indices = {}, match = @x
        bmodelica.matched_equation_instance %t1, indices = {}, match = @y
        bmodelica.matched_equation_instance %t2, indices = {}, match = @z
    }

    // CHECK:       bmodelica.initial {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t0]], indices = {}, match = @x
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @y
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t2]], indices = {}, match = @z
    // CHECK-NEXT:  }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t3, indices = {}, match = @z
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t3]], indices = {}, match = @z
    // CHECK-NEXT:  }
}

// -----

// CHECK-LABEL: @DynamicWithDependency

bmodelica.model @DynamicWithDependency {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // CHECK-DAG:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    // CHECK-DAG:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    // COM: x = 0
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: z = 0
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    // COM: z = y
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @z : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

    bmodelica.initial {
        bmodelica.matched_equation_instance %t0, indices = {}, match = @x
        bmodelica.matched_equation_instance %t1, indices = {}, match = @y
        bmodelica.matched_equation_instance %t2, indices = {}, match = @z
    }

    // CHECK:       bmodelica.initial {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t0]], indices = {}, match = @x
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t1]], indices = {}, match = @y
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t2]], indices = {}, match = @z
    // CHECK-NEXT:  }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t3, indices = {}, match = @z
    }

    // CHECK:       bmodelica.dynamic {
    // CHECK-DAG:       bmodelica.matched_equation_instance %[[t3]], indices = {}, match = @z
    // CHECK-NEXT:  }
}
