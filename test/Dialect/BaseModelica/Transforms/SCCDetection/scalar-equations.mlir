// RUN: modelica-opt %s --split-input-file --detect-scc --canonicalize-model-for-debug | FileCheck %s

// CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
// CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.scc {
// CHECK-NEXT:          bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:          bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.int>

    // x = y
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        %1 = bmodelica.variable_get @y : !bmodelica.int
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // y = 1 - x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @y : !bmodelica.int
        %1 = bmodelica.constant #bmodelica<int 1>
        %2 = bmodelica.variable_get @x : !bmodelica.int
        %3 = bmodelica.sub %1, %2 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    bmodelica.dynamic {
        bmodelica.matched_equation_instance %t0 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
        bmodelica.matched_equation_instance %t1 {path = #bmodelica<equation_path [L, 0]>} : !bmodelica.equation
    }
}
