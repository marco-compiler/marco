// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --canonicalize | FileCheck %s

// CHECK-DAG: %[[lowerBound:.*]] = modelica.constant 23 : index
// CHECK-DAG: %[[upperBound:.*]] = arith.constant 51 : index
// CHECK-DAG: %[[step:.*]] = modelica.constant 9 : index
// CHECK: scf.parallel (%{{.*}}) = (%[[lowerBound]]) to (%[[upperBound]]) step (%[[step]]) init (%{{.*}})

func.func @foo() -> !modelica.real {
    %begin = modelica.constant 23 : index
    %end = modelica.constant 57 : index
    %step = modelica.constant 9 : index
    %range = modelica.range %begin, %end, %step : (index, index, index) -> !modelica<range index>
    %array = modelica.constant #modelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !modelica.array<6x!modelica.real>

    %result = modelica.reduction "add", iterables = [%range], inductions = [%i0: index] {
        %element = modelica.load %array[%i0] : !modelica.array<6x!modelica.real>
        modelica.yield %element : !modelica.real
    } : (!modelica<range index>) -> !modelica.real

    func.return %result : !modelica.real
}

// -----

// CHECK-DAG:   %[[array:.*]] = modelica.constant #modelica.real_array<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : !modelica.array<2x3x!modelica.real>
// CHECK-DAG:   %[[init:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[load:.*]] = modelica.load %[[array]][%[[i0]], %[[i1]]]
// CHECK:           %[[load_casted:.*]] = builtin.unrealized_conversion_cast %[[load]] : !modelica.real to f64
// CHECK:           scf.reduce(%[[load_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.addf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index) -> !modelica.real {
    %0 = modelica.range %begin0, %end0, %step0 : (index, index, index) -> !modelica<range index>
    %1 = modelica.range %begin1, %end1, %step1 : (index, index, index) -> !modelica<range index>
    %2 = modelica.constant #modelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !modelica.array<2x3x!modelica.real>

    %3 = modelica.reduction "add", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = modelica.load %2[%i0, %i1] : !modelica.array<2x3x!modelica.real>
        modelica.yield %5 : !modelica.real
    } : (!modelica<range index>, !modelica<range index>) -> !modelica.real

    func.return %3 : !modelica.real
}

// -----

// CHECK-DAG:   %[[array:.*]] = modelica.constant #modelica.real_array<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : !modelica.array<2x3x!modelica.real>
// CHECK-DAG:   %[[init:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[load:.*]] = modelica.load %[[array]][%[[i0]], %[[i1]]]
// CHECK:           %[[load_casted:.*]] = builtin.unrealized_conversion_cast %[[load]] : !modelica.real to f64
// CHECK:           scf.reduce(%[[load_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.mulf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index) -> !modelica.real {
    %0 = modelica.range %begin0, %end0, %step0 : (index, index, index) -> !modelica<range index>
    %1 = modelica.range %begin1, %end1, %step1 : (index, index, index) -> !modelica<range index>
    %2 = modelica.constant #modelica.real_array<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]> : !modelica.array<2x3x!modelica.real>

    %3 = modelica.reduction "mul", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = modelica.load %2[%i0, %i1] : !modelica.array<2x3x!modelica.real>
        modelica.yield %5 : !modelica.real
    } : (!modelica<range index>, !modelica<range index>) -> !modelica.real

    func.return %3 : !modelica.real
}
