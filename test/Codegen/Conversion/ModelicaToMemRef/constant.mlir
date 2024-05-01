// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK: memref.global "private" constant @[[global:.*]] : memref<2xi1> = dense<[false, true]>

// CHECK-LABEL: @boolean1DArray
// CHECK: %[[global_get:.*]] = memref.get_global @[[global]] : memref<2xi1>
// CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %[[global_get]] : memref<2xi1> to !bmodelica.array<2x!bmodelica.bool>
// CHECK: return %[[cast]]

func.func @boolean1DArray() -> !bmodelica.array<2x!bmodelica.bool> {
    %0 = bmodelica.constant #bmodelica.bool_array<[false, true]> : !bmodelica.array<2x!bmodelica.bool>
    func.return %0 : !bmodelica.array<2x!bmodelica.bool>
}

// -----

// CHECK: memref.global "private" constant @[[global:.*]] : memref<2xi64> = dense<[1, 2]>

// CHECK-LABEL: @integer1DArray
// CHECK: %[[global_get:.*]] = memref.get_global @[[global]] : memref<2xi64>
// CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %[[global_get]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK: return %[[cast]]

func.func @integer1DArray() -> !bmodelica.array<2x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.int_array<[1, 2]> : !bmodelica.array<2x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x!bmodelica.int>
}

// -----

// CHECK: memref.global "private" constant @[[global:.*]] : memref<2xf64> = dense<[1.000000e+00, 2.000000e+00]>

// CHECK-LABEL: @real1DArray
// CHECK: %[[global_get:.*]] = memref.get_global @[[global]] : memref<2xf64>
// CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %[[global_get]] : memref<2xf64> to !bmodelica.array<2x!bmodelica.real>
// CHECK: return %[[cast]]

func.func @real1DArray() -> !bmodelica.array<2x!bmodelica.real> {
    %0 = bmodelica.constant #bmodelica.real_array<[1.0, 2.0]> : !bmodelica.array<2x!bmodelica.real>
    func.return %0 : !bmodelica.array<2x!bmodelica.real>
}

// -----

// CHECK: memref.global "private" constant @[[global:.*]] : memref<2x3x4xi64>
// CHECK-SAME{LITERAL}: = dense<[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]>

// CHECK-LABEL: @integer3DArray
// CHECK: %[[global_get:.*]] = memref.get_global @[[global]] : memref<2x3x4xi64>
// CHECK: %[[cast:.*]] = builtin.unrealized_conversion_cast %[[global_get]] : memref<2x3x4xi64> to !bmodelica.array<2x3x4x!bmodelica.int>
// CHECK: return %[[cast]]

func.func @integer3DArray() -> !bmodelica.array<2x3x4x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.int_array<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]> : !bmodelica.array<2x3x4x!bmodelica.int>
    func.return %0 : !bmodelica.array<2x3x4x!bmodelica.int>
}

// -----

// CHECK: memref.global "private" constant @[[global:.*]] : memref<2xi64> = dense<[1, 2]>

// CHECK-LABEL: @sameValue
// CHECK: %[[global_get_1:.*]] = memref.get_global @[[global]] : memref<2xi64>
// CHECK: %[[cast_1:.*]] = builtin.unrealized_conversion_cast %[[global_get_1]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK: %[[global_get_2:.*]] = memref.get_global @[[global]] : memref<2xi64>
// CHECK: %[[cast_2:.*]] = builtin.unrealized_conversion_cast %[[global_get_2]] : memref<2xi64> to !bmodelica.array<2x!bmodelica.int>
// CHECK: return %[[cast_1]], %[[cast_2]]

func.func @sameValue() -> (!bmodelica.array<2x!bmodelica.int>, !bmodelica.array<2x!bmodelica.int>) {
    %0 = bmodelica.constant #bmodelica.int_array<[1, 2]> : !bmodelica.array<2x!bmodelica.int>
    %1 = bmodelica.constant #bmodelica.int_array<[1, 2]> : !bmodelica.array<2x!bmodelica.int>
    func.return %0, %1 : !bmodelica.array<2x!bmodelica.int>, !bmodelica.array<2x!bmodelica.int>
}
