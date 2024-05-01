// RUN: modelica-opt %s --split-input-file --create-equation-templates --canonicalize | FileCheck %s

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }

// CHECK:       bmodelica.main_model {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.main_model {
        bmodelica.equation {
            %0 = bmodelica.variable_get @x : !bmodelica.int
            %1 = bmodelica.constant #bmodelica.int<0>
            %2 = bmodelica.equation_side %0 : tuple<!bmodelica.int>
            %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
            bmodelica.equation_sides %2, %3 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
        }
    }
}
