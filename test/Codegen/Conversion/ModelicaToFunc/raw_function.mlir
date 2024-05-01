// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK:       func.func @booleanScalarInput(%[[arg0:.*]]: i1) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @booleanScalarInput(%arg0: !bmodelica.bool) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @integerScalarInput(%[[arg0:.*]]: i64) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @integerScalarInput(%arg0: !bmodelica.int) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @realScalarInput(%[[arg0:.*]]: f64) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @realScalarInput(%arg0: !bmodelica.real) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @staticArrayInput(%[[arg0:.*]]: memref<5x3xi64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @staticArrayInput(%arg0: !bmodelica.array<5x3x!bmodelica.int>) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @dynamicArrayInput(%[[arg0:.*]]: memref<5x?xi64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @dynamicArrayInput(%arg0: !bmodelica.array<5x?x!bmodelica.int>) {
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @booleanScalarOutput() -> i1 {
// CHECK-NEXT:      %[[result:.*]] = bmodelica.constant #bmodelica.bool<true>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.bool to i1
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

bmodelica.raw_function @booleanScalarOutput() -> !bmodelica.bool {
    %0 = bmodelica.constant #bmodelica.bool<true>
    bmodelica.raw_return %0 : !bmodelica.bool
}

// -----

// CHECK:       func.func @integerScalarOutput() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.int to i64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

bmodelica.raw_function @integerScalarOutput() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica.int<0>
    bmodelica.raw_return %0 : !bmodelica.int
}

// -----

// CHECK:       func.func @realScalarOutput() -> f64 {
// CHECK-NEXT:      %[[result:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !bmodelica.real to f64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

bmodelica.raw_function @realScalarOutput() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica.real<0.0>
    bmodelica.raw_return %0 : !bmodelica.real
}

// -----

// CHECK:       func.func @staticArrayOutput() -> memref<3x5xi64> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.alloc : <3x5x!bmodelica.int>
// CHECK-NEXT:      %[[x_casted:.*]] = builtin.unrealized_conversion_cast %[[x]] : !bmodelica.array<3x5x!bmodelica.int> to memref<3x5xi64>
// CHECK-NEXT:      return %[[x_casted]]
// CHECK-NEXT:  }

bmodelica.raw_function @staticArrayOutput() -> !bmodelica.array<3x5x!bmodelica.int> {
    %0 = bmodelica.alloc : <3x5x!bmodelica.int>
    bmodelica.raw_return %0 : !bmodelica.array<3x5x!bmodelica.int>
}

// -----

// CHECK:       func.func @dynamicArrayOutput() {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[fakeArray:.*]] = bmodelica.alloc : <3x0x!bmodelica.int>
// CHECK-NEXT:      %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !bmodelica.array<3x0x!bmodelica.int> to memref<3x0xi64>
// CHECK-NEXT:      %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[fakeArray_casted_2]], %[[ptr]][]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @dynamicArrayOutput() {
    %0 = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @scalarOutputVariableGet() -> i64 {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.alloca : <!bmodelica.int>
// CHECK-NEXT:      %[[result:.*]] = bmodelica.load %[[x]][]
// CHECK-NEXT:      %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : !bmodelica.int to i64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

bmodelica.raw_function @scalarOutputVariableGet() -> !bmodelica.int {
    %0 = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.raw_variable_get %0 : !bmodelica.variable<!bmodelica.int, output>
    bmodelica.raw_return %1 : !bmodelica.int
}

// -----

// CHECK:       func.func @scalarOutputVariableSet() {
// CHECK-DAG:       %[[x:.*]] = bmodelica.alloca : <!bmodelica.int>
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      bmodelica.store %[[x]][], %[[value]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @scalarOutputVariableSet() {
    %0 = bmodelica.raw_variable : !bmodelica.variable<!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.constant #bmodelica.int<0>
    bmodelica.raw_variable_set %0, %1 : !bmodelica.variable<!bmodelica.int, output>, !bmodelica.int
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @staticArrayOutputVariableGet() -> memref<3x5xi64> {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.alloc : <3x5x!bmodelica.int>
// CHECK-NEXT:      %[[result:.*]] = builtin.unrealized_conversion_cast %[[x]] : !bmodelica.array<3x5x!bmodelica.int> to memref<3x5xi64>
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

bmodelica.raw_function @staticArrayOutputVariableGet() -> !bmodelica.array<3x5x!bmodelica.int> {
    %0 = bmodelica.raw_variable : !bmodelica.variable<3x5x!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.raw_variable_get %0 : !bmodelica.variable<3x5x!bmodelica.int, output>
    bmodelica.raw_return %1 : !bmodelica.array<3x5x!bmodelica.int>
}

// -----

// CHECK:       func.func @staticArrayOutputVariableSet() {
// CHECK-DAG:       %[[x:.*]] = bmodelica.alloc : <3x5x!bmodelica.int>
// CHECK-DAG:       %[[value:.*]] = bmodelica.alloca : <3x5x!bmodelica.int>
// CHECK-NEXT:      bmodelica.array_copy %[[value]], %[[x]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @staticArrayOutputVariableSet() {
    %0 = bmodelica.raw_variable : !bmodelica.variable<3x5x!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.alloca : <3x5x!bmodelica.int>
    bmodelica.raw_variable_set %0, %1 : !bmodelica.variable<3x5x!bmodelica.int, output>, !bmodelica.array<3x5x!bmodelica.int>
    bmodelica.raw_return
}

// -----

// CHECK:       func.func @dynamicArrayOutputGet() -> memref<3x?xi64> {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:           %[[result:.*]] = memref.load %[[ptr]][]
// CHECK:           return %[[result]]
// CHECK-NEXT:  }

bmodelica.raw_function @dynamicArrayOutputGet() -> !bmodelica.array<3x?x!bmodelica.int> {
    %0 = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.raw_variable_get %0 : !bmodelica.variable<3x?x!bmodelica.int, output>
    bmodelica.raw_return %1 : !bmodelica.array<3x?x!bmodelica.int>
}

// -----

// CHECK:       func.func @dynamicArrayOutputSet() {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:           %[[new:.*]] = bmodelica.alloc : <3x5x!bmodelica.int>
// CHECK-NEXT:      %[[previous:.*]] = memref.load %alloca[] : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[previous_casted:.*]] = builtin.unrealized_conversion_cast %[[previous]] : memref<3x?xi64> to !bmodelica.array<3x?x!bmodelica.int>
// CHECK-NEXT:      bmodelica.free %[[previous_casted]] : <3x?x!bmodelica.int>
// CHECK-NEXT:      %[[new_casted_1:.*]] = builtin.unrealized_conversion_cast %[[new]] : !bmodelica.array<3x5x!bmodelica.int> to memref<3x5xi64>
// CHECK-NEXT:      %[[new_casted_2:.*]] = memref.cast %[[new_casted_1]] : memref<3x5xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[new_casted_2]], %[[ptr]][] : memref<memref<3x?xi64>>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

bmodelica.raw_function @dynamicArrayOutputSet() {
    %0 = bmodelica.raw_variable : !bmodelica.variable<3x?x!bmodelica.int, output> {name = "x"}
    %1 = bmodelica.alloc : <3x5x!bmodelica.int>
    bmodelica.raw_variable_set %0, %1 : !bmodelica.variable<3x?x!bmodelica.int, output>, !bmodelica.array<3x5x!bmodelica.int>
    bmodelica.raw_return
}
