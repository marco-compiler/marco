// RUN: marco-opt %s --split-input-file | FileCheck %s

// Relational and logical operators.

// CHECK-LABEL: @relational
// CHECK: modelica.eq %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.bool
// CHECK: modelica.ne %{{.*}}, %{{.*}} : (!modelica.bool, !modelica.bool) -> !modelica.bool
// CHECK: modelica.lt %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.bool
// CHECK: modelica.le %{{.*}}, %{{.*}} : (!modelica.string, !modelica.string) -> !modelica.bool
// CHECK: modelica.gt %{{.*}}, %{{.*}} : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.bool
// CHECK: modelica.ge %{{.*}}, %{{.*}} : (!modelica.real, !modelica.real) -> !modelica.bool

func.func @relational(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: !modelica.bool, %arg3: !modelica.string, %arg4: !modelica.enum<@E>) {
    %0 = modelica.eq %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.bool
    %1 = modelica.ne %arg2, %arg2 : (!modelica.bool, !modelica.bool) -> !modelica.bool
    %2 = modelica.lt %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    %3 = modelica.le %arg3, %arg3 : (!modelica.string, !modelica.string) -> !modelica.bool
    %4 = modelica.gt %arg4, %arg4 : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.bool
    %5 = modelica.ge %arg1, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    return
}

// -----

// CHECK-LABEL: @logical
// CHECK: modelica.not %{{.*}} : (!modelica.bool) -> !modelica.bool
// CHECK: modelica.and %{{.*}}, %{{.*}} : (tensor<3x!modelica.bool>, tensor<3x!modelica.bool>) -> tensor<3x!modelica.bool>
// CHECK: modelica.or %{{.*}}, %{{.*}} : (!modelica.bool, !modelica.bool) -> !modelica.bool

func.func @logical(%arg0: !modelica.bool, %arg1: tensor<3x!modelica.bool>) {
    %0 = modelica.not %arg0 : (!modelica.bool) -> !modelica.bool
    %1 = modelica.and %arg1, %arg1 : (tensor<3x!modelica.bool>, tensor<3x!modelica.bool>) -> tensor<3x!modelica.bool>
    %2 = modelica.or %arg0, %arg0 : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return
}

// -----

// If-expressions, with implicit conversion of the
// Integer branch.

// CHECK-LABEL: @ifExpr
// CHECK:       %[[result:.*]] = modelica.if_expr %{{.*}} : !modelica.real {
// CHECK-NEXT:      modelica.yield %{{.*}} : !modelica.real
// CHECK-NEXT:  } else {
// CHECK-NEXT:      %[[nested:.*]] = modelica.if_expr %{{.*}} : !modelica.int {
// CHECK-NEXT:          modelica.yield %{{.*}} : !modelica.int
// CHECK-NEXT:      } else {
// CHECK-NEXT:          modelica.yield %{{.*}} : !modelica.int
// CHECK-NEXT:      }
// CHECK-NEXT:      modelica.yield %[[nested]] : !modelica.int
// CHECK-NEXT:  }

func.func @ifExpr(%arg0: !modelica.bool, %arg1: !modelica.real, %arg2: !modelica.int) {
    %0 = modelica.if_expr %arg0 : !modelica.real {
        modelica.yield %arg1 : !modelica.real
    } else {
        %1 = modelica.if_expr %arg0 : !modelica.int {
            modelica.yield %arg2 : !modelica.int
        } else {
            modelica.yield %arg2 : !modelica.int
        }
        modelica.yield %1 : !modelica.int
    }
    return
}
