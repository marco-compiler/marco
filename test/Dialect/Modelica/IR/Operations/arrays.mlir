// RUN: marco-opt %s --split-input-file | FileCheck %s

// Array constructors and functions.

// CHECK-LABEL: @range
// CHECK: modelica.range %{{.*}} to %{{.*}} : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
// CHECK: modelica.range %{{.*}} to %{{.*}} step %{{.*}} : (!modelica.int, !modelica.real, !modelica.real) -> tensor<?x!modelica.real>
// CHECK: modelica.range %{{.*}} to %{{.*}} : (!modelica.enum<@E>, !modelica.enum<@E>) -> tensor<?x!modelica.enum<@E>>

func.func @range(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: !modelica.enum<@E>) {
    %0 = modelica.range %arg0 to %arg0 : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
    %1 = modelica.range %arg0 to %arg1 step %arg1 : (!modelica.int, !modelica.real, !modelica.real) -> tensor<?x!modelica.real>
    %2 = modelica.range %arg2 to %arg2 : (!modelica.enum<@E>, !modelica.enum<@E>) -> tensor<?x!modelica.enum<@E>>
    return
}

// -----

// CHECK-LABEL: @array
// CHECK: modelica.array %{{.*}}, %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>
// CHECK: modelica.array %{{.*}}, %{{.*}} : (tensor<2x!modelica.int>, tensor<2x!modelica.int>) -> tensor<2x2x!modelica.int>
// CHECK: modelica.array %{{.*}}, %{{.*}} : (!modelica.object<@R>, !modelica.object<@R>) -> tensor<2x!modelica.object<@R>>

func.func @array(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<2x!modelica.int>, %arg3: !modelica.object<@R>) {
    %0 = modelica.array %arg0, %arg1, %arg0 : (!modelica.int, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>
    %1 = modelica.array %arg2, %arg2 : (tensor<2x!modelica.int>, tensor<2x!modelica.int>) -> tensor<2x2x!modelica.int>
    %2 = modelica.array %arg3, %arg3 : (!modelica.object<@R>, !modelica.object<@R>) -> tensor<2x!modelica.object<@R>>
    return
}

// -----

// CHECK-LABEL: @concatenation
// CHECK: modelica.cat %{{.*}}, %{{.*}}, %{{.*}} : (!modelica.int, tensor<2x3x!modelica.real>, tensor<2x?x!modelica.int>) -> tensor<2x?x!modelica.real>
// CHECK: modelica.promote %{{.*}}, 2 : (tensor<3x!modelica.real>) -> tensor<3x1x!modelica.real>

func.func @concatenation(%arg0: tensor<2x3x!modelica.real>, %arg1: tensor<2x?x!modelica.int>, %arg2: tensor<3x!modelica.real>) {
    %dim = modelica.constant #modelica<int 1> : !modelica.int
    %0 = modelica.cat %dim, %arg0, %arg1 : (!modelica.int, tensor<2x3x!modelica.real>, tensor<2x?x!modelica.int>) -> tensor<2x?x!modelica.real>
    %1 = modelica.promote %arg2, 2 : (tensor<3x!modelica.real>) -> tensor<3x1x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @constructorFunctions
// CHECK: modelica.identity %{{.*}} : (!modelica.int) -> tensor<3x3x!modelica.int>
// CHECK: modelica.diagonal %{{.*}} : (tensor<?x!modelica.real>) -> tensor<?x?x!modelica.real>
// CHECK: modelica.zeros %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> tensor<3x?x!modelica.int>
// CHECK: modelica.ones %{{.*}} : (!modelica.int) -> tensor<3x!modelica.int>
// CHECK: modelica.fill %{{.*}}, %{{.*}} : (tensor<2x!modelica.bool>, !modelica.int) -> tensor<3x2x!modelica.bool>
// CHECK: modelica.linspace %{{.*}}, %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>

func.func @constructorFunctions(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<?x!modelica.real>, %arg3: tensor<2x!modelica.bool>) {
    %n = modelica.constant #modelica<int 3> : !modelica.int
    %0 = modelica.identity %n : (!modelica.int) -> tensor<3x3x!modelica.int>
    %1 = modelica.diagonal %arg2 : (tensor<?x!modelica.real>) -> tensor<?x?x!modelica.real>
    %2 = modelica.zeros %n, %arg0 : (!modelica.int, !modelica.int) -> tensor<3x?x!modelica.int>
    %3 = modelica.ones %n : (!modelica.int) -> tensor<3x!modelica.int>
    %4 = modelica.fill %arg3, %n : (tensor<2x!modelica.bool>, !modelica.int) -> tensor<3x2x!modelica.bool>
    %5 = modelica.linspace %arg0, %arg1, %n : (!modelica.int, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @dimensions
// CHECK: modelica.ndims %{{.*}} : (tensor<2x3x!modelica.real>) -> !modelica.int
// CHECK: modelica.size %{{.*}} : (tensor<2x3x!modelica.real>) -> tensor<2x!modelica.int>
// CHECK: modelica.size %{{.*}}, %{{.*}} : (tensor<2x3x!modelica.real>, !modelica.int) -> !modelica.int
// CHECK: modelica.scalar %{{.*}} : (tensor<1x1x!modelica.real>) -> !modelica.real
// CHECK: modelica.vector %{{.*}} : (tensor<1x3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.matrix %{{.*}} : (tensor<3x!modelica.real>) -> tensor<3x1x!modelica.real>

func.func @dimensions(%arg0: tensor<2x3x!modelica.real>, %arg1: !modelica.int, %arg2: tensor<1x1x!modelica.real>, %arg3: tensor<1x3x!modelica.real>, %arg4: tensor<3x!modelica.real>) {
    %0 = modelica.ndims %arg0 : (tensor<2x3x!modelica.real>) -> !modelica.int
    %1 = modelica.size %arg0 : (tensor<2x3x!modelica.real>) -> tensor<2x!modelica.int>
    %2 = modelica.size %arg0, %arg1 : (tensor<2x3x!modelica.real>, !modelica.int) -> !modelica.int
    %3 = modelica.scalar %arg2 : (tensor<1x1x!modelica.real>) -> !modelica.real
    %4 = modelica.vector %arg3 : (tensor<1x3x!modelica.real>) -> tensor<3x!modelica.real>
    %5 = modelica.matrix %arg4 : (tensor<3x!modelica.real>) -> tensor<3x1x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @arrayReduce
// CHECK: modelica.array_reduce sum %{{.*}} : (tensor<3x!modelica.int>) -> !modelica.int
// CHECK: modelica.array_reduce product %{{.*}} : (tensor<2x2x!modelica.real>) -> !modelica.real
// CHECK: modelica.array_reduce min %{{.*}} : (tensor<?x!modelica.enum<@E>>) -> !modelica.enum<@E>
// CHECK: modelica.array_reduce max %{{.*}} : (tensor<3x!modelica.int>) -> !modelica.int

func.func @arrayReduce(%arg0: tensor<3x!modelica.int>, %arg1: tensor<2x2x!modelica.real>, %arg2: tensor<?x!modelica.enum<@E>>) {
    %0 = modelica.array_reduce sum %arg0 : (tensor<3x!modelica.int>) -> !modelica.int
    %1 = modelica.array_reduce product %arg1 : (tensor<2x2x!modelica.real>) -> !modelica.real
    %2 = modelica.array_reduce min %arg2 : (tensor<?x!modelica.enum<@E>>) -> !modelica.enum<@E>
    %3 = modelica.array_reduce max %arg0 : (tensor<3x!modelica.int>) -> !modelica.int
    return
}

// -----

// CHECK-LABEL: @algebra
// CHECK: modelica.transpose %{{.*}} : (tensor<2x3x!modelica.int>) -> tensor<3x2x!modelica.int>
// CHECK: modelica.outer_product %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, tensor<2x!modelica.int>) -> tensor<3x2x!modelica.real>
// CHECK: modelica.symmetric %{{.*}} : (tensor<3x3x!modelica.real>) -> tensor<3x3x!modelica.real>
// CHECK: modelica.cross %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.skew %{{.*}} : (tensor<3x!modelica.real>) -> tensor<3x3x!modelica.real>

func.func @algebra(%arg0: tensor<2x3x!modelica.int>, %arg1: tensor<3x!modelica.real>, %arg2: tensor<2x!modelica.int>, %arg3: tensor<3x3x!modelica.real>) {
    %0 = modelica.transpose %arg0 : (tensor<2x3x!modelica.int>) -> tensor<3x2x!modelica.int>
    %1 = modelica.outer_product %arg1, %arg2 : (tensor<3x!modelica.real>, tensor<2x!modelica.int>) -> tensor<3x2x!modelica.real>
    %2 = modelica.symmetric %arg3 : (tensor<3x3x!modelica.real>) -> tensor<3x3x!modelica.real>
    %3 = modelica.cross %arg1, %arg1 : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %4 = modelica.skew %arg1 : (tensor<3x!modelica.real>) -> tensor<3x3x!modelica.real>
    return
}
