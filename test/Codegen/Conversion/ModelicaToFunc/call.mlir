// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

modelica.raw_function @foo(%arg0: !modelica.bool) -> !modelica.bool {
    modelica.raw_return %arg0 : !modelica.bool
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.bool) -> !modelica.bool
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.bool to i1
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (i1) -> i1
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i1 to !modelica.bool
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !modelica.bool) -> !modelica.bool {
    %0 = modelica.call @foo(%arg0) : (!modelica.bool) -> !modelica.bool
    return %0 : !modelica.bool
}

// -----

modelica.raw_function @foo(%arg0: !modelica.int) -> !modelica.int {
    modelica.raw_return %arg0 : !modelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.int) -> !modelica.int
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (i64) -> i64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !modelica.int
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !modelica.int) -> !modelica.int {
    %0 = modelica.call @foo(%arg0) : (!modelica.int) -> !modelica.int
    return %0 : !modelica.int
}

// -----

modelica.raw_function @foo(%arg0: !modelica.real) -> !modelica.real {
    modelica.raw_return %arg0 : !modelica.real
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.real) -> !modelica.real
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (f64) -> f64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !modelica.real
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !modelica.real) -> !modelica.real {
    %0 = modelica.call @foo(%arg0) : (!modelica.real) -> !modelica.real
    return %0 : !modelica.real
}

// -----

modelica.raw_function @foo(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.array<5x3x!modelica.int> {
    modelica.raw_return %arg0 : !modelica.array<5x3x!modelica.int>
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !modelica.array<5x3x!modelica.int>) -> !modelica.array<5x3x!modelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.array<5x3x!modelica.int> to memref<5x3xi64>
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (memref<5x3xi64>) -> memref<5x3xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<5x3xi64> to !modelica.array<5x3x!modelica.int>
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !modelica.array<5x3x!modelica.int>) -> !modelica.array<5x3x!modelica.int> {
    %0 = modelica.call @foo(%arg0) : (!modelica.array<5x3x!modelica.int>) -> !modelica.array<5x3x!modelica.int>
    return %0 : !modelica.array<5x3x!modelica.int>
}
