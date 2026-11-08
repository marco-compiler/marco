// RUN: marco-opt %s --split-input-file | FileCheck %s

// Arithmetic operators.

// CHECK-LABEL: @neg
// CHECK: modelica.neg %{{.*}} : (!modelica.int) -> !modelica.int
// CHECK: modelica.neg %{{.*}} : (tensor<3x!modelica.real>) -> tensor<3x!modelica.real>

func.func @neg(%arg0: !modelica.int, %arg1: tensor<3x!modelica.real>) {
    %0 = modelica.neg %arg0 : (!modelica.int) -> !modelica.int
    %1 = modelica.neg %arg1 : (tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @add
// CHECK: modelica.add %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK: modelica.add %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.real
// CHECK: modelica.add %{{.*}}, %{{.*}} : (tensor<3x!modelica.int>, tensor<?x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.add %{{.*}}, %{{.*}} : (!modelica.string, !modelica.string) -> !modelica.string

func.func @add(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<3x!modelica.int>, %arg3: tensor<?x!modelica.real>, %arg4: !modelica.string) {
    %0 = modelica.add %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.add %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.real
    %2 = modelica.add %arg2, %arg3 : (tensor<3x!modelica.int>, tensor<?x!modelica.real>) -> tensor<3x!modelica.real>
    %3 = modelica.add %arg4, %arg4 : (!modelica.string, !modelica.string) -> !modelica.string
    return
}

// -----

// CHECK-LABEL: @sub
// CHECK: modelica.sub %{{.*}}, %{{.*}} : (tensor<2x3x!modelica.real>, tensor<2x3x!modelica.real>) -> tensor<2x3x!modelica.real>

func.func @sub(%arg0: tensor<2x3x!modelica.real>) {
    %0 = modelica.sub %arg0, %arg0 : (tensor<2x3x!modelica.real>, tensor<2x3x!modelica.real>) -> tensor<2x3x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @mul
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> !modelica.real
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, tensor<3x4x!modelica.real>) -> tensor<4x!modelica.real>
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (tensor<4x3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<4x!modelica.real>
// CHECK: modelica.mul %{{.*}}, %{{.*}} : (tensor<4x3x!modelica.real>, tensor<3x4x!modelica.real>) -> tensor<4x4x!modelica.real>

func.func @mul(%arg0: !modelica.int, %arg1: tensor<3x!modelica.real>, %arg2: tensor<3x4x!modelica.real>, %arg3: tensor<4x3x!modelica.real>) {
    %0 = modelica.mul %arg0, %arg1 : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %1 = modelica.mul %arg1, %arg1 : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> !modelica.real
    %2 = modelica.mul %arg1, %arg2 : (tensor<3x!modelica.real>, tensor<3x4x!modelica.real>) -> tensor<4x!modelica.real>
    %3 = modelica.mul %arg3, %arg1 : (tensor<4x3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<4x!modelica.real>
    %4 = modelica.mul %arg3, %arg2 : (tensor<4x3x!modelica.real>, tensor<3x4x!modelica.real>) -> tensor<4x4x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @div
// CHECK: modelica.div %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.real
// CHECK: modelica.div %{{.*}}, %{{.*}} : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.real>

func.func @div(%arg0: !modelica.int, %arg1: tensor<3x!modelica.int>) {
    %0 = modelica.div %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.real
    %1 = modelica.div %arg1, %arg0 : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @pow
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.real
// CHECK: modelica.pow %{{.*}}, %{{.*}} : (tensor<3x3x!modelica.int>, !modelica.int) -> tensor<3x3x!modelica.int>

func.func @pow(%arg0: !modelica.int, %arg1: tensor<3x3x!modelica.int>) {
    %0 = modelica.pow %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.real
    %1 = modelica.pow %arg1, %arg0 : (tensor<3x3x!modelica.int>, !modelica.int) -> tensor<3x3x!modelica.int>
    return
}

// -----

// CHECK-LABEL: @elementwise
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.add_ew %{{.*}}, %{{.*}} : (tensor<3x!modelica.string>, !modelica.string) -> tensor<3x!modelica.string>
// CHECK: modelica.sub_ew %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, !modelica.int) -> tensor<3x!modelica.real>
// CHECK: modelica.mul_ew %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.div_ew %{{.*}}, %{{.*}} : (!modelica.int, tensor<3x!modelica.int>) -> tensor<3x!modelica.real>
// CHECK: modelica.pow_ew %{{.*}}, %{{.*}} : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.real>

func.func @elementwise(%arg0: !modelica.int, %arg1: tensor<3x!modelica.real>, %arg2: tensor<3x!modelica.string>, %arg3: !modelica.string, %arg4: tensor<3x!modelica.int>) {
    %0 = modelica.add_ew %arg0, %arg1 : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %1 = modelica.add_ew %arg2, %arg3 : (tensor<3x!modelica.string>, !modelica.string) -> tensor<3x!modelica.string>
    %2 = modelica.sub_ew %arg1, %arg0 : (tensor<3x!modelica.real>, !modelica.int) -> tensor<3x!modelica.real>
    %3 = modelica.mul_ew %arg1, %arg1 : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %4 = modelica.div_ew %arg0, %arg4 : (!modelica.int, tensor<3x!modelica.int>) -> tensor<3x!modelica.real>
    %5 = modelica.pow_ew %arg4, %arg0 : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.real>
    return
}
