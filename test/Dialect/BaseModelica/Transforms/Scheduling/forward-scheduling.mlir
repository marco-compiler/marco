// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// COM: for i in 1:9 loop
// COM:   x[i] = x[i - 1]
// COM:
// COM: x[0] = 0

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // x[i] = x[i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %1 = bmodelica.tensor_extract %0[%i0] : tensor<10x!bmodelica.int>
        %2 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant 1 : index
        %4 = bmodelica.sub %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %2[%4] : tensor<10x!bmodelica.int>
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.int>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}] attributes {id = "t0"}

    // x[0] = 0
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.constant 0 : index
        %1 = bmodelica.variable_get @x : tensor<10x!bmodelica.int>
        %2 = bmodelica.tensor_extract %1[%0] : tensor<10x!bmodelica.int>
        %3 = bmodelica.constant #bmodelica<int 0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.int>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.equation_instance %t0, indices = {[1,9]}, match = <@x, {[1,9]}>
            }
            bmodelica.scc {
                bmodelica.equation_instance %t1, match = <@x, {[0,0]}>
            }

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t1]]
            // CHECK-SAME:      match = <@x, {[0,0]}>
            // CHECK-NEXT:  }
            // CHECK-NEXT:  bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
            // CHECK-SAME:      indices = {[1,9]}
            // CHECK-SAME:      match = <@x, {[1,9]}>
            // CHECK-SAME:      schedule = [forward]
            // CHECK-NEXT:  }
        }
    }
}