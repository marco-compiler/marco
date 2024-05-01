// RUN: modelica-opt %s --split-input-file --convert-modelica-to-memref | FileCheck %s

// CHECK-LABEL: @staticArrays
// CHECK: %[[source:.*]] = memref.alloc() : memref<3x2xi64>
// CHECK: %[[destination:.*]] = memref.alloc() : memref<3x2xi64>
// CHECK: memref.copy %[[source]], %[[destination]]

func.func @staticArrays() {
    %0 = bmodelica.alloc : <3x2x!bmodelica.int>
    %1 = bmodelica.alloc : <3x2x!bmodelica.int>
    bmodelica.array_copy %0, %1 : !bmodelica.array<3x2x!bmodelica.int>, !bmodelica.array<3x2x!bmodelica.int>
    func.return
}

// -----

// CHECK-LABEL: @dynamicArrays
// CHECK: %[[source:.*]] = memref.alloc({{.*}}, {{.*}}) : memref<?x?xi64>
// CHECK: %[[destination:.*]] = memref.alloc({{.*}}, {{.*}}) : memref<?x?xi64>
// CHECK: memref.copy %[[source]], %[[destination]]

func.func @dynamicArrays() {
    %0 = arith.constant 2 : index
    %1 = bmodelica.alloc %0, %0 : <?x?x!bmodelica.int>
    %2 = bmodelica.alloc %0, %0 : <?x?x!bmodelica.int>
    bmodelica.array_copy %1, %2 : !bmodelica.array<?x?x!bmodelica.int>, !bmodelica.array<?x?x!bmodelica.int>
    func.return
}

// -----

// CHECK-LABEL: @implicitCast
// CHECK: %[[source:.*]] = memref.alloc() : memref<3x2xi64>
// CHECK: %[[destination:.*]] = memref.alloc() : memref<3x2xf64>

// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[d0:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[d1:.*]] = arith.constant 2 : index

// CHECK:   scf.for %[[i0:.*]] = %[[zero]] to %[[d0]] step %[[one]] {
// CHECK:       scf.for %[[i1:.*]] = %[[zero]] to %[[d1]] step %[[one]] {
// CHECK:           %[[value:.*]] = memref.load %[[source]][%[[i0]], %[[i1]]]
// CHECK:           %[[value_casted_1:.*]] = builtin.unrealized_conversion_cast %[[value]] : i64 to !bmodelica.int
// CHECK:           %[[value_casted_2:.*]] = bmodelica.cast %[[value_casted_1]] : !bmodelica.int -> !bmodelica.real
// CHECK:           %[[value_casted_3:.*]] = builtin.unrealized_conversion_cast %[[value_casted_2]] : !bmodelica.real to f64
// CHECK:           memref.store %[[value_casted_3]], %[[destination]][%[[i0]], %[[i1]]]
// CHECK:       }
// CHECK:   }

func.func @implicitCast() {
    %0 = bmodelica.alloc : <3x2x!bmodelica.int>
    %1 = bmodelica.alloc : <3x2x!bmodelica.real>
    bmodelica.array_copy %0, %1 : !bmodelica.array<3x2x!bmodelica.int>, !bmodelica.array<3x2x!bmodelica.real>
    func.return
}
