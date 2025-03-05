// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @p : !bmodelica.variable<!bmodelica.real, parameter>
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>

    // COM: x = p
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @p : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: x = 1
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0> : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: y = 20
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 20.0> : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    // COM: p = y
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @p : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

    bmodelica.schedule @ic {
        bmodelica.initial {
            bmodelica.start_equation_instance %t0
            bmodelica.scc {
                bmodelica.equation_instance %t1, match = @x
            }
            bmodelica.scc {
                bmodelica.equation_instance %t2, match = @y
            }
            bmodelica.scc {
                bmodelica.equation_instance %t3, match = @p
            }

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t2]]
            // CHECK-SAME:      match = @y
            // CHECK-NEXT:  }

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t3]]
            // CHECK-SAME:      match = @p
            // CHECK-NEXT:  }

            // CHECK:       bmodelica.start_equation_instance %[[t0]]

            // CHECK-NEXT:  bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t1]]
            // CHECK-SAME:      match = @x
            // CHECK-NEXT:  }
        }
    }

    // CHECK-NOT: bmodelica.start_equation_instance
    // CHECK-NOT: bmodelica.equation_instance
}
