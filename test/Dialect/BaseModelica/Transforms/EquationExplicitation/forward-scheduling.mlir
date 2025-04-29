// RUN: modelica-opt %s --split-input-file --equation-explicitation | FileCheck %s

// CHECK:       bmodelica.schedule @schedule {
// CHECK-NEXT:      bmodelica.dynamic {
// CHECK-NEXT:          bmodelica.schedule_block writtenVariables = [<@x, {[1,9]}>], readVariables = [<@x, {[0,8]}>] {
// CHECK-NEXT:              bmodelica.equation_call @[[eq:.*]] {[1,9]}
// CHECK-NEXT:          } {parallelizable = true}
// CHECK-NEXT:      }
// CHECK-NEXT:  }

// CHECK:       bmodelica.equation_function @[[eq]]
// CHECK-SAME:  (%[[lb:.*]]: index, %[[ub:.*]]: index)
// CHECK:       %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.for %[[i0:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:           bmodelica.constant 1 : index
// CHECK:           %[[var:.*]] = bmodelica.qualified_variable_get @Test::@x
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %[[var]][%[[i0]]]
// CHECK:           bmodelica.store %[[subscription]][], %{{.*}}
// CHECK:       }
// CHECK-NEXT:  bmodelica.yield

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<10x!bmodelica.int>

    // COM: x[i] = x[i - 1]
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

    bmodelica.schedule @schedule {
        bmodelica.dynamic {
            bmodelica.scc {
                bmodelica.equation_instance %t0, indices = {[1,9]}, match = <@x, {[1,9]}>, schedule = [forward]
            }
        }
    }
}
