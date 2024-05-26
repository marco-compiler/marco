// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --canonicalize | FileCheck %s

// CHECK-LABEL: @foo
// CHECK-DAG: %[[lowerBound:.*]] = bmodelica.constant 23 : index
// CHECK-DAG: %[[upperBound:.*]] = arith.constant 51 : index
// CHECK-DAG: %[[step:.*]] = bmodelica.constant 9 : index
// CHECK: scf.parallel (%{{.*}}) = (%[[lowerBound]]) to (%[[upperBound]]) step (%[[step]]) init (%{{.*}})

func.func @foo(%tensor: tensor<6x!bmodelica.real>) -> !bmodelica.real {
    %begin = bmodelica.constant 23 : index
    %end = bmodelica.constant 57 : index
    %step = bmodelica.constant 9 : index
    %range = bmodelica.range %begin, %end, %step : (index, index, index) -> !bmodelica<range index>

    %result = bmodelica.reduction "add", iterables = [%range], inductions = [%i0: index] {
        %element = bmodelica.tensor_extract %tensor[%i0] : tensor<6x!bmodelica.real>
        bmodelica.yield %element : !bmodelica.real
    } : (!bmodelica<range index>) -> !bmodelica.real

    func.return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME: (%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %[[tensor:.*]]: tensor<2x3x!bmodelica.real>)
// CHECK:       %[[init:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[extract:.*]] = bmodelica.tensor_extract %[[tensor]][%[[i0]], %[[i1]]]
// CHECK:           %[[extract_casted:.*]] = builtin.unrealized_conversion_cast %[[extract]] : !bmodelica.real to f64
// CHECK:           scf.reduce(%[[extract_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.addf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index,
               %tensor: tensor<2x3x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.range %begin0, %end0, %step0 : (index, index, index) -> !bmodelica<range index>
    %1 = bmodelica.range %begin1, %end1, %step1 : (index, index, index) -> !bmodelica<range index>

    %3 = bmodelica.reduction "add", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = bmodelica.tensor_extract %tensor[%i0, %i1] : tensor<2x3x!bmodelica.real>
        bmodelica.yield %5 : !bmodelica.real
    } : (!bmodelica<range index>, !bmodelica<range index>) -> !bmodelica.real

    func.return %3 : !bmodelica.real
}

// -----

// CHECK-LABEL: @foo
// CHECK-SAME:  (%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %[[tensor:.*]]: tensor<2x3x!bmodelica.real>)
// CHECK:       %[[init:.*]] = arith.constant 1.000000e+00 : f64
// CHECK:       %[[result:.*]] = scf.parallel (%[[i0:.*]], %[[i1:.*]]) = (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) step (%{{.*}}, %{{.*}}) init (%[[init]]) -> f64 {
// CHECK:           %[[extract:.*]] = bmodelica.tensor_extract %[[tensor]][%[[i0]], %[[i1]]]
// CHECK:           %[[extract_casted:.*]] = builtin.unrealized_conversion_cast %[[extract]] : !bmodelica.real to f64
// CHECK:           scf.reduce(%[[extract_casted]] : f64) {
// CHECK:           ^{{.*}}(%[[first:.*]]: f64, %[[second:.*]]: f64):
// CHECK:               %[[reduced:.*]] = arith.mulf %[[first]], %[[second]]
// CHECK:               scf.reduce.return %[[reduced]]
// CHECK:           }
// CHECK:       }

func.func @foo(%begin0: index, %end0: index, %step0: index,
               %begin1: index, %end1: index, %step1: index,
               %tensor: tensor<2x3x!bmodelica.real>) -> !bmodelica.real {
    %0 = bmodelica.range %begin0, %end0, %step0 : (index, index, index) -> !bmodelica<range index>
    %1 = bmodelica.range %begin1, %end1, %step1 : (index, index, index) -> !bmodelica<range index>

    %3 = bmodelica.reduction "mul", iterables = [%0, %1], inductions = [%i0: index, %i1: index] {
        %5 = bmodelica.tensor_extract %tensor[%i0, %i1] : tensor<2x3x!bmodelica.real>
        bmodelica.yield %5 : !bmodelica.real
    } : (!bmodelica<range index>, !bmodelica<range index>) -> !bmodelica.real

    func.return %3 : !bmodelica.real
}
