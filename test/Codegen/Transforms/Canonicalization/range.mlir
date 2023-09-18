// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[range:.*]]: modelica.constant_range {lowerBound = 0 : index, step = 1 : index, upperBound = 5 : index}
// CHECK: return %[[range]] : !modelica<iterable index>

func.func @test() -> !modelica<iterable index> {
    %0 = modelica.constant 0 : index
    %1 = modelica.constant 5 : index
    %2 = modelica.constant 1 : index
    %3 = modelica.range %0, %1, %2 : (index, index, index) -> !modelica<iterable index>
    return  %3 : !modelica<iterable index>
}
