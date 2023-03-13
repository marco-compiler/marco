// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

// CHECK:       func.func @booleanScalarInput(%[[arg0:.*]]: i1) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @booleanScalarInput(%arg0: !modelica.bool) {
    modelica.raw_return
}

// -----

// CHECK:       func.func @integerScalarInput(%[[arg0:.*]]: i64) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @integerScalarInput(%arg0: !modelica.int) {
    modelica.raw_return
}

// -----

// CHECK:       func.func @realScalarInput(%[[arg0:.*]]: f64) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @realScalarInput(%arg0: !modelica.real) {
    modelica.raw_return
}

// -----

// CHECK:       func.func @staticArrayInput(%[[arg0:.*]]: memref<5x3xi64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @staticArrayInput(%arg0: !modelica.array<5x3x!modelica.int>) {
    modelica.raw_return
}

// -----

// CHECK:       func.func @dynamicArrayInput(%[[arg0:.*]]: memref<5x?xi64>) {
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @dynamicArrayInput(%arg0: !modelica.array<5x?x!modelica.int>) {
    modelica.raw_return
}

// -----

// CHECK:       func.func @booleanScalarOutput() -> i1 {
// CHECK-NEXT:      %[[result:.*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.bool to i1
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

modelica.raw_function @booleanScalarOutput() -> !modelica.bool {
    %0 = modelica.constant #modelica.bool<true>
    modelica.raw_return %0 : !modelica.bool
}

// -----

// CHECK:       func.func @integerScalarOutput() -> i64 {
// CHECK-NEXT:      %[[result:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.int to i64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

modelica.raw_function @integerScalarOutput() -> !modelica.int {
    %0 = modelica.constant #modelica.int<0>
    modelica.raw_return %0 : !modelica.int
}

// -----

// CHECK:       func.func @realScalarOutput() -> f64 {
// CHECK-NEXT:      %[[result:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : !modelica.real to f64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

modelica.raw_function @realScalarOutput() -> !modelica.real {
    %0 = modelica.constant #modelica.real<0.0>
    modelica.raw_return %0 : !modelica.real
}

// -----

// CHECK:       func.func @staticArrayOutput() -> memref<3x5xi64> {
// CHECK-NEXT:      %[[x:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK-NEXT:      %[[x_casted:.*]] = builtin.unrealized_conversion_cast %[[x]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-NEXT:      return %[[x_casted]]
// CHECK-NEXT:  }

modelica.raw_function @staticArrayOutput() -> !modelica.array<3x5x!modelica.int> {
    %0 = modelica.alloc : !modelica.array<3x5x!modelica.int>
    modelica.raw_return %0 : !modelica.array<3x5x!modelica.int>
}

// -----

// CHECK:       func.func @dynamicArrayOutput() {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[fakeArray:.*]] = modelica.alloc : !modelica.array<3x0x!modelica.int>
// CHECK-NEXT:      %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !modelica.array<3x0x!modelica.int> to memref<3x0xi64>
// CHECK-NEXT:      %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[fakeArray_casted_2]], %[[ptr]][]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @dynamicArrayOutput() {
    %0 = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
    modelica.raw_return
}

// -----

// CHECK:       func.func @scalarOutputVariableGet() -> i64 {
// CHECK-NEXT:      %[[x:.*]] = modelica.alloc : !modelica.array<!modelica.int>
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[x]][]
// CHECK-NEXT:      %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : !modelica.int to i64
// CHECK-NEXT:      return %[[result_casted]]
// CHECK-NEXT:  }

modelica.raw_function @scalarOutputVariableGet() -> !modelica.int {
    %0 = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
    %1 = modelica.raw_variable_get %0 : !modelica.variable<!modelica.int, output>
    modelica.raw_return %1 : !modelica.int
}

// -----

// CHECK:       func.func @scalarOutputVariableSet() {
// CHECK-DAG:       %[[x:.*]] = modelica.alloc : !modelica.array<!modelica.int>
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK-NEXT:      modelica.store %[[x]][], %[[value]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @scalarOutputVariableSet() {
    %0 = modelica.raw_variable : !modelica.variable<!modelica.int, output> {name = "x"}
    %1 = modelica.constant #modelica.int<0>
    modelica.raw_variable_set %0, %1 : !modelica.variable<!modelica.int, output>, !modelica.int
    modelica.raw_return
}

// -----

// CHECK:       func.func @staticArrayOutputVariableGet() -> memref<3x5xi64> {
// CHECK-NEXT:      %[[x:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK-NEXT:      %[[result:.*]] = builtin.unrealized_conversion_cast %[[x]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-NEXT:      return %[[result]]
// CHECK-NEXT:  }

modelica.raw_function @staticArrayOutputVariableGet() -> !modelica.array<3x5x!modelica.int> {
    %0 = modelica.raw_variable : !modelica.variable<3x5x!modelica.int, output> {name = "x"}
    %1 = modelica.raw_variable_get %0 : !modelica.variable<3x5x!modelica.int, output>
    modelica.raw_return %1 : !modelica.array<3x5x!modelica.int>
}

// -----

// CHECK:       func.func @staticArrayOutputVariableSet() {
// CHECK-DAG:       %[[x:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK-DAG:       %[[value:.*]] = modelica.alloca : !modelica.array<3x5x!modelica.int>
// CHECK-NEXT:      modelica.array_copy %[[value]], %[[x]]
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @staticArrayOutputVariableSet() {
    %0 = modelica.raw_variable : !modelica.variable<3x5x!modelica.int, output> {name = "x"}
    %1 = modelica.alloca : !modelica.array<3x5x!modelica.int>
    modelica.raw_variable_set %0, %1 : !modelica.variable<3x5x!modelica.int, output>, !modelica.array<3x5x!modelica.int>
    modelica.raw_return
}

// -----

// CHECK:       func.func @dynamicArrayOutputGet() -> memref<3x?xi64> {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:           %[[result:.*]] = memref.load %[[ptr]][]
// CHECK:           return %[[result]]
// CHECK-NEXT:  }

modelica.raw_function @dynamicArrayOutputGet() -> !modelica.array<3x?x!modelica.int> {
    %0 = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
    %1 = modelica.raw_variable_get %0 : !modelica.variable<3x?x!modelica.int, output>
    modelica.raw_return %1 : !modelica.array<3x?x!modelica.int>
}

// -----

// CHECK:       func.func @dynamicArrayOutputSet() {
// CHECK-NEXT:      %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:           %[[new:.*]] = modelica.alloc : !modelica.array<3x5x!modelica.int>
// CHECK-NEXT:      %[[previous:.*]] = memref.load %alloca[] : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[previous_casted:.*]] = builtin.unrealized_conversion_cast %[[previous]] : memref<3x?xi64> to !modelica.array<3x?x!modelica.int>
// CHECK-NEXT:      modelica.free %[[previous_casted]] : !modelica.array<3x?x!modelica.int>
// CHECK-NEXT:      %[[new_casted_1:.*]] = builtin.unrealized_conversion_cast %[[new]] : !modelica.array<3x5x!modelica.int> to memref<3x5xi64>
// CHECK-NEXT:      %[[new_casted_2:.*]] = memref.cast %[[new_casted_1]] : memref<3x5xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[new_casted_2]], %[[ptr]][] : memref<memref<3x?xi64>>
// CHECK-NEXT:      return
// CHECK-NEXT:  }

modelica.raw_function @dynamicArrayOutputSet() {
    %0 = modelica.raw_variable : !modelica.variable<3x?x!modelica.int, output> {name = "x"}
    %1 = modelica.alloc : !modelica.array<3x5x!modelica.int>
    modelica.raw_variable_set %0, %1 : !modelica.variable<3x?x!modelica.int, output>, !modelica.array<3x5x!modelica.int>
    modelica.raw_return
}
