// RUN: marco-opt %s --split-input-file | FileCheck %s

// Iterator-based expressions.

// The last iterator varies along the outermost dimension: the result of
// {i - j for i in 1:3, j in 1:2} has shape 2x3.

// CHECK-LABEL: @arrayFor
// CHECK:       modelica.array_for (%[[i:.*]] in %{{.*}} : tensor<3x!modelica.int>, %[[j:.*]] in %{{.*}} : tensor<2x!modelica.int>) -> tensor<2x3x!modelica.int> {
// CHECK-NEXT:      %[[sub:.*]] = modelica.sub %[[i]], %[[j]]
// CHECK-NEXT:      modelica.yield %[[sub]] : !modelica.int
// CHECK-NEXT:  }

func.func @arrayFor(%arg0: tensor<3x!modelica.int>, %arg1: tensor<2x!modelica.int>) {
    %0 = modelica.array_for (%i in %arg0 : tensor<3x!modelica.int>, %j in %arg1 : tensor<2x!modelica.int>) -> tensor<2x3x!modelica.int> {
        %1 = modelica.sub %i, %j : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %1 : !modelica.int
    }
    return
}

// -----

// Array elements and Real results.

// CHECK-LABEL: @arrayForArrayElements
// CHECK: modelica.array_for (%{{.*}} in %{{.*}} : tensor<?x!modelica.real>) -> tensor<?x2x!modelica.real>

func.func @arrayForArrayElements(%arg0: tensor<?x!modelica.real>, %arg1: tensor<2x!modelica.int>) {
    %0 = modelica.array_for (%i in %arg0 : tensor<?x!modelica.real>) -> tensor<?x2x!modelica.real> {
        modelica.yield %arg1 : tensor<2x!modelica.int>
    }
    return
}

// -----

// CHECK-LABEL: @reduction
// CHECK:       modelica.reduction sum (%[[i:.*]] in %{{.*}} : tensor<10x!modelica.int>) -> !modelica.int {
// CHECK-NEXT:      %[[mul:.*]] = modelica.mul %[[i]], %[[i]]
// CHECK-NEXT:      modelica.yield %[[mul]] : !modelica.int
// CHECK-NEXT:  }
// CHECK: modelica.reduction product (%{{.*}} in %{{.*}} : tensor<10x!modelica.int>, %{{.*}} in %{{.*}} : tensor<10x!modelica.int>) -> !modelica.real
// CHECK: modelica.reduction min (%{{.*}} in %{{.*}} : tensor<?x!modelica.enum<@E>>) -> !modelica.enum<@E>
// CHECK: modelica.reduction max (%{{.*}} in %{{.*}} : tensor<10x!modelica.int>) -> !modelica.int
// CHECK: modelica.reduction sum (%{{.*}} in %{{.*}} : tensor<10x!modelica.int>) -> tensor<2x!modelica.real>

func.func @reduction(%arg0: tensor<10x!modelica.int>, %arg1: tensor<?x!modelica.enum<@E>>, %arg2: tensor<2x!modelica.real>) {
    %0 = modelica.reduction sum (%i in %arg0 : tensor<10x!modelica.int>) -> !modelica.int {
        %1 = modelica.mul %i, %i : (!modelica.int, !modelica.int) -> !modelica.int
        modelica.yield %1 : !modelica.int
    }
    %2 = modelica.reduction product (%i in %arg0 : tensor<10x!modelica.int>, %j in %arg0 : tensor<10x!modelica.int>) -> !modelica.real {
        modelica.yield %i : !modelica.int
    }
    %3 = modelica.reduction min (%e in %arg1 : tensor<?x!modelica.enum<@E>>) -> !modelica.enum<@E> {
        modelica.yield %e : !modelica.enum<@E>
    }
    %4 = modelica.reduction max (%i in %arg0 : tensor<10x!modelica.int>) -> !modelica.int {
        modelica.yield %i : !modelica.int
    }
    %5 = modelica.reduction sum (%i in %arg0 : tensor<10x!modelica.int>) -> tensor<2x!modelica.real> {
        modelica.yield %arg2 : tensor<2x!modelica.real>
    }
    return
}
