// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       bmodelica.function @M1_M2_foo {
// CHECK-NEXT:      bmodelica.algorithm {
// CHECK-NEXT:          bmodelica.call @M1_M2_foo() : () -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:  }

bmodelica.model @M1 {
    bmodelica.model @M2 {
        bmodelica.function @foo {
            bmodelica.algorithm {
                bmodelica.call @M1::@M2::@foo() : () -> ()
            }
        }
    }
}
