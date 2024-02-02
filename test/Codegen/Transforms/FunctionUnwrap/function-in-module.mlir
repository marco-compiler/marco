// RUN: modelica-opt %s --split-input-file --function-unwrap | FileCheck %s

// CHECK:       modelica.model @M1 {
// CHECK-NEXT:      modelica.model @M2 {
// CHECK-NEXT:          modelica.algorithm {
// CHECK-NEXT:              modelica.call @foo() : () -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:      }
// CHECK-NEXT:  }
// CHECK-NEXT:  modelica.function @foo {
// CHECK-NEXT:  }

module {
    modelica.model @M1 {
        modelica.model @M2 {
            modelica.algorithm {
                modelica.call @foo() : () -> ()
            }
        }
    }

    modelica.function @foo {

    }
}
