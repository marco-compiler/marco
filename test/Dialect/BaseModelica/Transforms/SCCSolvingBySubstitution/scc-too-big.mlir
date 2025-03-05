// RUN: modelica-opt %s --split-input-file --scc-solving-substitution="max-equations-in-scc=3" | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
// CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.scc {
// CHECK-NEXT:          bmodelica.equation_instance %[[t0]], match = @x1
// CHECK-NEXT:          bmodelica.equation_instance %[[t1]], match = @x2
// CHECK-NEXT:          bmodelica.equation_instance %[[t2]], match = @x3
// CHECK-NEXT:          bmodelica.equation_instance %[[t3]], match = @x4
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x1 : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @x3 : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @x4 : !bmodelica.variable<!bmodelica.int>

    // x1 = x2
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x1 : !bmodelica.int
        %1 = bmodelica.variable_get @x2 : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x2 = x3
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @x2 : !bmodelica.int
        %1 = bmodelica.variable_get @x3 : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x3 = x4
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @x3 : !bmodelica.int
        %1 = bmodelica.variable_get @x4 : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // x4 = 1 - x1
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @x4 : !bmodelica.int
        %1 = bmodelica.variable_get @x1 : !bmodelica.int
        %2 = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
        %3 = bmodelica.sub %2, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.scc {
            bmodelica.equation_instance %t0, match = @x1
            bmodelica.equation_instance %t1, match = @x2
            bmodelica.equation_instance %t2, match = @x3
            bmodelica.equation_instance %t3, match = @x4
        }
    }
}
