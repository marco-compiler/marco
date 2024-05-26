// RUN: modelica-opt %s --split-input-file --create-equation-templates --canonicalize | FileCheck %s

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[extract:.*]] = bmodelica.tensor_extract %[[x]][%[[i0]]]
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[extract]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
// CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.dynamic {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.dynamic {
        bmodelica.for_equation %i = 0 to 2 {
            bmodelica.equation {
                %0 = bmodelica.variable_get @x : tensor<3x!bmodelica.int>
                %1 = bmodelica.tensor_extract %0[%i] : tensor<3x!bmodelica.int>
                %2 = bmodelica.constant #bmodelica<int 0>
                %3 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
                %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
                bmodelica.equation_sides %3, %4 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}
