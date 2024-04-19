// RUN: marco -S --omc-bypass -o - %s | FileCheck %s

// CHECK:       _modelName:
// CHECK-NEXT:      .asciz "M"

modelica.model @M {

}
