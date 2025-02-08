// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       bmodelica.model @M1 {
// CHECK-NEXT:  }
// CHECK-NEXT:  bmodelica.function @M1_foo {
// CHECK-NEXT:      bmodelica.algorithm {
// CHECK-NEXT:          bmodelica.call @M1_foo_bar()
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  bmodelica.function @M1_foo_bar

bmodelica.model @M1 {
    bmodelica.function @foo {
        bmodelica.function @bar {

        }

        bmodelica.algorithm {
            bmodelica.call @M1::@foo::@bar() : () -> ()
        }
    }
}
