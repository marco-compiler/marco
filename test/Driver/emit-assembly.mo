// RUN: marco -S --omc-bypass -o - %s | FileCheck %s

// CHECK:       _modelName:
// CHECK-NEXT:      .asciz "M"

model M
end M;
