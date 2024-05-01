// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --canonicalize | FileCheck %s

// CHECK-DAG: %[[lowerBound:.*]] = bmodelica.constant 23 : index
// CHECK-DAG: %[[upperBound:.*]] = arith.constant 51 : index
// CHECK-DAG: %[[step:.*]] = bmodelica.constant 9 : index
// CHECK: scf.parallel (%{{.*}}) = (%[[lowerBound]]) to (%[[upperBound]]) step (%[[step]]) init (%{{.*}})

func.func @foo() -> !bmodelica.real {
    %begin = bmodelica.constant 23 : index
    %end = bmodelica.constant 57 : index
    %step = bmodelica.constant 9 : index
    %range = bmodelica.range %begin, %end, %step : (index, index, index) -> !bmodelica<range index>
    %array = bmodelica.constant #bmodelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !bmodelica.array<6x!bmodelica.real>

    %result = bmodelica.reduction "add", iterables = [%range], inductions = [%i0: index] {
        %element = bmodelica.load %array[%i0] : !bmodelica.array<6x!bmodelica.real>
        bmodelica.yield %element : !bmodelica.real
    } : (!bmodelica<range index>) -> !bmodelica.real

    func.return %result : !bmodelica.real
}

// -----

// CHECK-DAG:   %[[array:.*]] = bmodelica.constant #bmodelica.real_array<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : !bmodelica.array<2x3x!bmodelica.real>
// CHECK-DAG:   %[[init:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[load:.*]] = bmodelica.load %[[array]][%[[i0]], %[[i1]]]
// CHECK:           %[[load_casted:.*]] = builtin.unrealized_conversion_cast %[[load]] : !bmodelica.real to f64
// CHECK:           scf.reduce(%[[load_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.addf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index) -> !bmodelica.real {
    %0 = bmodelica.range %begin0, %end0, %step0 : (index, index, index) -> !bmodelica<range index>
    %1 = bmodelica.range %begin1, %end1, %step1 : (index, index, index) -> !bmodelica<range index>
    %2 = bmodelica.constant #bmodelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !bmodelica.array<2x3x!bmodelica.real>

    %3 = bmodelica.reduction "add", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = bmodelica.load %2[%i0, %i1] : !bmodelica.array<2x3x!bmodelica.real>
        bmodelica.yield %5 : !bmodelica.real
    } : (!bmodelica<range index>, !bmodelica<range index>) -> !bmodelica.real

    func.return %3 : !bmodelica.real
}

// -----

// CHECK-DAG:   %[[array:.*]] = bmodelica.constant #bmodelica.real_array<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : !bmodelica.array<2x3x!bmodelica.real>
// CHECK-DAG:   %[[init:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[load:.*]] = bmodelica.load %[[array]][%[[i0]], %[[i1]]]
// CHECK:           %[[load_casted:.*]] = builtin.unrealized_conversion_cast %[[load]] : !bmodelica.real to f64
// CHECK:           scf.reduce(%[[load_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.mulf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index) -> !bmodelica.real {
    %0 = bmodelica.range %begin0, %end0, %step0 : (index, index, index) -> !bmodelica<range index>
    %1 = bmodelica.range %begin1, %end1, %step1 : (index, index, index) -> !bmodelica<range index>
    %2 = bmodelica.constant #bmodelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !bmodelica.array<2x3x!bmodelica.real>

    %3 = bmodelica.reduction "mul", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = bmodelica.load %2[%i0, %i1] : !bmodelica.array<2x3x!bmodelica.real>
        bmodelica.yield %5 : !bmodelica.real
    } : (!bmodelica<range index>, !bmodelica<range index>) -> !bmodelica.real

    func.return %3 : !bmodelica.real
}
