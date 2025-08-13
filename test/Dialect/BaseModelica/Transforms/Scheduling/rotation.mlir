// RUN: modelica-opt %s --split-input-file --schedule | FileCheck %s

// COM: for i in 1:9 loop
// COM:   for j in 1:9 loop
// COM:     x[j + 1, i - 2] = x[j, i - 1]

// CHECK-LABEL: @Test

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<20x20x!bmodelica.int>

    // x[j + 1, i - 2] = x[j, i - 1]
    %t0 = bmodelica.equation_template inductions = [%i0, %i1] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @x : tensor<20x20x!bmodelica.int>
        %1 = bmodelica.constant 1 : index
        %2 = bmodelica.add %i1, %1 : (index, index) -> index
        %3 = bmodelica.constant 2 : index
        %4 = bmodelica.sub %i0, %3 : (index, index) -> index
        %5 = bmodelica.tensor_extract %0[%2, %4] : tensor<20x20x!bmodelica.int>
        %6 = bmodelica.sub %i0, %1 : (index, index) -> index
        %7 = bmodelica.tensor_extract %0[%i1, %6] : tensor<20x20x!bmodelica.int>
        %8 = bmodelica.equation_side %5 : tuple<!bmodelica.int>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.int>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.int>, tuple<!bmodelica.int>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [{{.*}}, {{.*}}] attributes {id = "t0"}

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.equation_instance %t0, indices = {[3,5][9,10]}, match = <@x, {[10,11][1,3]}>
            }

            // CHECK:       bmodelica.scc {
            // CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
            // CHECK-SAME:      indices = {[3,5][9,10]}
            // CHECK-SAME:      match = <@x, {[10,11][1,3]}>
            // CHECK-SAME:      schedule = [backward, forward]
            // CHECK-NEXT:  }
        }
    }
}