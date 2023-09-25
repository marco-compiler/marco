// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK: %[[range:.*]] = modelica.constant #modelica.int_range<0, 5, 1>
// CHECK: return %[[range]] : !modelica<range index>

func.func @test() -> !modelica<range index> {
    %0 = modelica.constant 0 : index
    %1 = modelica.constant 5 : index
    %2 = modelica.constant 1 : index
    %3 = modelica.range %0, %1, %2 : (index, index, index) -> !modelica<range index>
    return  %3 : !modelica<range index>
}
