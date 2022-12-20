// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false})" | FileCheck %s

// Scalar variable

// CHECK:       modelica.raw_function @scalarVariable() -> !modelica.int {
// CHECK:           %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[y]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:      modelica.raw_return %[[result]]
// CHECK-NEXT:  }

modelica.function @scalarVariable : () -> (!modelica.int) {
    %0 = modelica.member_create @y : !modelica.member<!modelica.int, output>
}

// -----

// Load of a scalar variable

// CHECK:   modelica.raw_function @scalarVariableLoad() -> !modelica.int {
// CHECK:       %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK:       %{{.*}} = modelica.load %[[y]][]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @scalarVariableLoad : () -> (!modelica.int) {
    %0 = modelica.member_create @y : !modelica.member<!modelica.int, output>
    %1 = modelica.member_load %0 : !modelica.member<!modelica.int, output>
}

// -----

// Store of a scalar variable

// CHECK:       modelica.raw_function @scalarVariableStore() -> !modelica.int {
// CHECK-DAG:       %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.store %[[y]][], %[[value]]
// CHECK:           cf.br ^{{.*}}
// CHECK:       }

modelica.function @scalarVariableStore : () -> (!modelica.int) {
    %0 = modelica.member_create @y : !modelica.member<!modelica.int, output>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<!modelica.int, output>, !modelica.int
}

// -----

// Static array

// CHECK:       modelica.raw_function @staticArray() -> !modelica.array<3x2x!modelica.int> {
// CHECK:           %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return %[[y]]
// CHECK-NEXT:  }

modelica.function @staticArray : () -> (!modelica.array<3x2x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int, output>
}

// -----

// Load of a static array

// CHECK:   modelica.raw_function @staticArrayLoad() -> !modelica.array<3x2x!modelica.int> {
// CHECK:       %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       %{{.*}} = modelica.load %[[y]][{{.*}}, {{.*}}]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @staticArrayLoad : () -> (!modelica.array<3x2x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int, output>
    %1 = modelica.member_load %0 : !modelica.member<3x2x!modelica.int, output>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
}

// -----

// Store of a static array

// CHECK:   modelica.raw_function @staticArrayStore() -> !modelica.array<3x2x!modelica.int> {
// CHECK:       %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       %[[value:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       modelica.array_copy %[[value]], %[[y]]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @staticArrayStore : () -> (!modelica.array<3x2x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int, output>
    %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
    modelica.member_store %0, %1 : !modelica.member<3x2x!modelica.int, output>, !modelica.array<3x2x!modelica.int>
}

// -----

// Dynamic array

// CHECK:       modelica.raw_function @dynamicArray() -> !modelica.array<3x?x!modelica.int> {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[fakeArray:.*]] = modelica.alloc : !modelica.array<3x0x!modelica.int>
// CHECK-NEXT:      %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !modelica.array<3x0x!modelica.int> to memref<3x0xi64>
// CHECK-NEXT:      %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[fakeArray_casted_2]], %[[alloca]][]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[result:.*]] = memref.load %[[alloca]]
// CHECK-NEXT:      %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<3x?xi64> to !modelica.array<3x?x!modelica.int>
// CHECK-NEXT:      modelica.raw_return %[[result_casted]]
// CHECK-NEXT:  }

modelica.function @dynamicArray : () -> (!modelica.array<3x?x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int, output>
}

// -----

// Load of a dynamic array

// CHECK:   modelica.raw_function @dynamicArrayLoad() -> !modelica.array<3x?x!modelica.int> {
// CHECK:       %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:       %[[fakeArray:.*]] = modelica.alloc : !modelica.array<3x0x!modelica.int>
// CHECK:       %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !modelica.array<3x0x!modelica.int> to memref<3x0xi64>
// CHECK:       %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK:       memref.store %[[fakeArray_casted_2]], %[[ptr]][]
// CHECK:       %[[array:.*]] = memref.load %[[ptr]][]
// CHECK:       %[[array_casted:.*]] = builtin.unrealized_conversion_cast %[[array]] : memref<3x?xi64> to !modelica.array<3x?x!modelica.int>
// CHECK:       {{.*}} = modelica.load %[[array_casted]][%{{.*}}, %{{.*}}]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @dynamicArrayLoad : () -> (!modelica.array<3x?x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int, output>
    %1 = modelica.member_load %0 : !modelica.member<3x?x!modelica.int, output>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
}

// -----

// Store of a dynamic array

// CHECK:   modelica.raw_function @dynamicArrayStore() -> !modelica.array<3x?x!modelica.int> {
// CHECK:       %[[ptr:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK:       %[[fakeArray:.*]] = modelica.alloc : !modelica.array<3x0x!modelica.int>
// CHECK:       %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !modelica.array<3x0x!modelica.int> to memref<3x0xi64>
// CHECK:       %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK:       memref.store %[[fakeArray_casted_2]], %[[ptr]][]
// CHECK:       %[[value:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       %[[previousArray:.*]] = memref.load %[[ptr]][]
// CHECK:       %[[previousArray_casted:.*]] = builtin.unrealized_conversion_cast %[[previousArray]] : memref<3x?xi64> to !modelica.array<3x?x!modelica.int>
// CHECK:       modelica.free %[[previousArray_casted]]
// CHECK:       %[[value_casted_1:.*]] = builtin.unrealized_conversion_cast %[[value]] : !modelica.array<3x2x!modelica.int> to memref<3x2xi64>
// CHECK:       %[[value_casted_2:.*]] = memref.cast %[[value_casted_1]] : memref<3x2xi64> to memref<3x?xi64>
// CHECK:       memref.store %[[value_casted_2]], %[[ptr]][]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @dynamicArrayStore : () -> (!modelica.array<3x?x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int, output>
    %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
    modelica.member_store %0, %1 : !modelica.member<3x?x!modelica.int, output>, !modelica.array<3x2x!modelica.int>
}
