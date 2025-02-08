// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       bmodelica.model @M1 {
// CHECK-NEXT:      bmodelica.model @M2 {
// CHECK-NEXT:          bmodelica.dynamic {
// CHECK-NEXT:              bmodelica.algorithm {
// CHECK-NEXT:                  bmodelica.call @M1_M2_foo
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  bmodelica.function @M1_M2_foo

bmodelica.model @M1 {
    bmodelica.model @M2 {
        bmodelica.function @foo {

        }

        bmodelica.dynamic {
            bmodelica.algorithm {
                bmodelica.call @M1::@M2::@foo() : () -> ()
            }
        }
    }
}
