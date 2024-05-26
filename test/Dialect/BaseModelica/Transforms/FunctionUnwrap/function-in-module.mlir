// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       bmodelica.model @M1 {
// CHECK-NEXT:      bmodelica.model @M2 {
// CHECK-NEXT:          bmodelica.dynamic {
// CHECK-NEXT:              bmodelica.algorithm {
// CHECK-NEXT:                  bmodelica.call @foo() : () -> ()
// CHECK-NEXT:              }
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  bmodelica.function @foo {
// CHECK-NEXT:  }

module {
    bmodelica.model @M1 {
        bmodelica.model @M2 {
            bmodelica.dynamic {
                bmodelica.algorithm {
                    bmodelica.call @foo() : () -> ()
                }
            }
        }
    }

    bmodelica.function @foo {

    }
}
