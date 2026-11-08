// RUN: marco-opt %s --split-input-file | FileCheck %s

// Built-in operators and functions.

// CHECK-LABEL: @numeric
// CHECK: modelica.abs %{{.*}} : (!modelica.int) -> !modelica.int
// CHECK: modelica.sign %{{.*}} : (tensor<3x!modelica.real>) -> tensor<3x!modelica.int>
// CHECK: modelica.sqrt %{{.*}} : (!modelica.int) -> !modelica.real
// CHECK: modelica.nth_root %{{.*}}, %{{.*}} : (!modelica.real, !modelica.int) -> !modelica.real
// CHECK: modelica.min %{{.*}}, %{{.*}} : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.max %{{.*}}, %{{.*}} : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.enum<@E>

func.func @numeric(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<3x!modelica.real>, %arg3: !modelica.enum<@E>) {
    %0 = modelica.abs %arg0 : (!modelica.int) -> !modelica.int
    %1 = modelica.sign %arg2 : (tensor<3x!modelica.real>) -> tensor<3x!modelica.int>
    %2 = modelica.sqrt %arg0 : (!modelica.int) -> !modelica.real
    %3 = modelica.nth_root %arg1, %arg0 : (!modelica.real, !modelica.int) -> !modelica.real
    %4 = modelica.min %arg0, %arg2 : (!modelica.int, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %5 = modelica.max %arg3, %arg3 : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.enum<@E>
    return
}

// -----

// CHECK-LABEL: @conversions
// CHECK: modelica.enum_to_integer %{{.*}} : (!modelica.enum<@E>) -> !modelica.int
// CHECK: modelica.integer_to_enum %{{.*}} : (tensor<2x!modelica.int>) -> tensor<2x!modelica.enum<@E>>
// CHECK: modelica.to_string %{{.*}} : (!modelica.bool) -> !modelica.string
// CHECK: modelica.to_string %{{.*}} minimumLength %{{.*}} leftJustified %{{.*}} significantDigits %{{.*}} : (!modelica.real, !modelica.int, !modelica.bool, !modelica.int) -> !modelica.string
// CHECK: modelica.to_string %{{.*}} format %{{.*}} : (!modelica.int, !modelica.string) -> !modelica.string

func.func @conversions(%arg0: !modelica.enum<@E>, %arg1: tensor<2x!modelica.int>, %arg2: !modelica.bool, %arg3: !modelica.real, %arg4: !modelica.int, %arg5: !modelica.string) {
    %0 = modelica.enum_to_integer %arg0 : (!modelica.enum<@E>) -> !modelica.int
    %1 = modelica.integer_to_enum %arg1 : (tensor<2x!modelica.int>) -> tensor<2x!modelica.enum<@E>>
    %2 = modelica.to_string %arg2 : (!modelica.bool) -> !modelica.string
    %3 = modelica.to_string %arg3 minimumLength %arg4 leftJustified %arg2 significantDigits %arg4 : (!modelica.real, !modelica.int, !modelica.bool, !modelica.int) -> !modelica.string
    %4 = modelica.to_string %arg4 format %arg5 : (!modelica.int, !modelica.string) -> !modelica.string
    return
}

// -----

// CHECK-LABEL: @eventTriggering
// CHECK: modelica.div_trunc %{{.*}}, %{{.*}} : (!modelica.int, !modelica.int) -> !modelica.int
// CHECK: modelica.mod %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.real
// CHECK: modelica.rem %{{.*}}, %{{.*}} : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.int>
// CHECK: modelica.ceil %{{.*}} : (!modelica.real) -> !modelica.real
// CHECK: modelica.floor %{{.*}} : (!modelica.real) -> !modelica.real
// CHECK: modelica.integer %{{.*}} : (!modelica.real) -> !modelica.int
// CHECK: modelica.delay %{{.*}}, %{{.*}} : (!modelica.real, !modelica.real) -> !modelica.real
// CHECK: modelica.delay %{{.*}}, %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>

func.func @eventTriggering(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<3x!modelica.int>, %arg3: tensor<3x!modelica.real>) {
    %0 = modelica.div_trunc %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.mod %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.real
    %2 = modelica.rem %arg2, %arg0 : (tensor<3x!modelica.int>, !modelica.int) -> tensor<3x!modelica.int>
    %3 = modelica.ceil %arg1 : (!modelica.real) -> !modelica.real
    %4 = modelica.floor %arg1 : (!modelica.real) -> !modelica.real
    %5 = modelica.integer %arg1 : (!modelica.real) -> !modelica.int
    %6 = modelica.delay %arg1, %arg1 : (!modelica.real, !modelica.real) -> !modelica.real
    %7 = modelica.delay %arg3, %arg1, %arg0 : (tensor<3x!modelica.real>, !modelica.real, !modelica.int) -> tensor<3x!modelica.real>
    return
}

// -----

// CHECK-LABEL: @elementary
// CHECK: modelica.sin %{{.*}} : (!modelica.int) -> !modelica.real
// CHECK: modelica.cos %{{.*}} : (tensor<2x!modelica.real>) -> tensor<2x!modelica.real>
// CHECK: modelica.tan
// CHECK: modelica.asin
// CHECK: modelica.acos
// CHECK: modelica.atan %
// CHECK: modelica.sinh
// CHECK: modelica.cosh
// CHECK: modelica.tanh
// CHECK: modelica.exp
// CHECK: modelica.log %
// CHECK: modelica.log10
// CHECK: modelica.atan2 %{{.*}}, %{{.*}} : (!modelica.real, !modelica.int) -> !modelica.real

func.func @elementary(%arg0: !modelica.int, %arg1: !modelica.real, %arg2: tensor<2x!modelica.real>) {
    %0 = modelica.sin %arg0 : (!modelica.int) -> !modelica.real
    %1 = modelica.cos %arg2 : (tensor<2x!modelica.real>) -> tensor<2x!modelica.real>
    %2 = modelica.tan %arg1 : (!modelica.real) -> !modelica.real
    %3 = modelica.asin %arg1 : (!modelica.real) -> !modelica.real
    %4 = modelica.acos %arg1 : (!modelica.real) -> !modelica.real
    %5 = modelica.atan %arg1 : (!modelica.real) -> !modelica.real
    %6 = modelica.sinh %arg1 : (!modelica.real) -> !modelica.real
    %7 = modelica.cosh %arg1 : (!modelica.real) -> !modelica.real
    %8 = modelica.tanh %arg1 : (!modelica.real) -> !modelica.real
    %9 = modelica.exp %arg1 : (!modelica.real) -> !modelica.real
    %10 = modelica.log %arg1 : (!modelica.real) -> !modelica.real
    %11 = modelica.log10 %arg1 : (!modelica.real) -> !modelica.real
    %12 = modelica.atan2 %arg1, %arg0 : (!modelica.real, !modelica.int) -> !modelica.real
    return
}

// -----

// CHECK-LABEL: @special
// CHECK: modelica.der %{{.*}} : (tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
// CHECK: modelica.homotopy %{{.*}}, %{{.*}} : (!modelica.real, !modelica.int) -> !modelica.real
// CHECK: modelica.semi_linear %{{.*}}, %{{.*}}, %{{.*}} : (tensor<3x!modelica.real>, !modelica.real, !modelica.real) -> tensor<3x!modelica.real>
// CHECK: modelica.time : !modelica.real

func.func @special(%arg0: tensor<3x!modelica.real>, %arg1: !modelica.real, %arg2: !modelica.int) {
    %0 = modelica.der %arg0 : (tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    %1 = modelica.homotopy %arg1, %arg2 : (!modelica.real, !modelica.int) -> !modelica.real
    %2 = modelica.semi_linear %arg0, %arg1, %arg1 : (tensor<3x!modelica.real>, !modelica.real, !modelica.real) -> tensor<3x!modelica.real>
    %3 = modelica.time : !modelica.real
    return
}

// -----

// CHECK-LABEL: @eventRelated
// CHECK: modelica.initial : !modelica.bool
// CHECK: modelica.terminal : !modelica.bool
// CHECK: modelica.no_event %{{.*}} : (!modelica.bool) -> !modelica.bool
// CHECK: modelica.smooth %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.real
// CHECK: modelica.sample %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real) -> !modelica.bool
// CHECK: modelica.pre %{{.*}} : (tensor<3x!modelica.int>) -> tensor<3x!modelica.int>
// CHECK: modelica.edge %{{.*}} : (!modelica.bool) -> !modelica.bool
// CHECK: modelica.change %{{.*}} : (tensor<3x!modelica.int>) -> tensor<3x!modelica.bool>

func.func @eventRelated(%arg0: !modelica.bool, %arg1: !modelica.int, %arg2: !modelica.real, %arg3: tensor<3x!modelica.int>) {
    %0 = modelica.initial : !modelica.bool
    %1 = modelica.terminal : !modelica.bool
    %2 = modelica.no_event %arg0 : (!modelica.bool) -> !modelica.bool
    %3 = modelica.smooth %arg1, %arg2 : (!modelica.int, !modelica.real) -> !modelica.real
    %4 = modelica.sample %arg1, %arg2 : (!modelica.int, !modelica.real) -> !modelica.bool
    %5 = modelica.pre %arg3 : (tensor<3x!modelica.int>) -> tensor<3x!modelica.int>
    %6 = modelica.edge %arg0 : (!modelica.bool) -> !modelica.bool
    %7 = modelica.change %arg3 : (tensor<3x!modelica.int>) -> tensor<3x!modelica.bool>
    return
}
