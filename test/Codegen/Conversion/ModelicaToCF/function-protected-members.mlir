// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false})" | FileCheck %s

// Scalar variable

// CHECK:       modelica.raw_function @scalarVariable() {
// CHECK:           %[[x:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @scalarVariable : () -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
}

// -----

// Load of a scalar variable

// CHECK:   modelica.raw_function @scalarVariableLoad() {
// CHECK:       %[[x:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK:       %{{.*}} = modelica.load %[[x]][]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @scalarVariableLoad : () -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.member_load %0 : !modelica.member<!modelica.int>
}

// -----

// Store of a scalar variable

// CHECK:       modelica.raw_function @scalarVariableStore() {
// CHECK-DAG:       %[[x:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-DAG:       %[[value:.*]] = modelica.constant #modelica.int<0>
// CHECK:           modelica.store %[[x]][], %[[value]]
// CHECK:           cf.br ^{{.*}}
// CHECK:       }

modelica.function @scalarVariableStore : () -> () {
    %0 = modelica.member_create @x : !modelica.member<!modelica.int>
    %1 = modelica.constant #modelica.int<0>
    modelica.member_store %0, %1 : !modelica.member<!modelica.int>, !modelica.int
}

// -----

// Static array

// CHECK:       modelica.raw_function @staticArray() {
// CHECK:           %[[x:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.free %[[x]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @staticArray : () -> () {
    %0 = modelica.member_create @x : !modelica.member<3x2x!modelica.int>
}

// -----

// Load of a static array

// CHECK:   modelica.raw_function @staticArrayLoad() {
// CHECK:       %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       %{{.*}} = modelica.load %[[y]][{{.*}}, {{.*}}]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @staticArrayLoad : () -> () {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.member<3x2x!modelica.int>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x2x!modelica.int>
}

// -----

// Store of a static array

// CHECK:   modelica.raw_function @staticArrayStore() {
// CHECK:       %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       %[[value:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK:       modelica.array_copy %[[value]], %[[y]]
// CHECK:       cf.br ^{{.*}}
// CHECK:   }

modelica.function @staticArrayStore : () -> () {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int>
    %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
    modelica.member_store %0, %1 : !modelica.member<3x2x!modelica.int>, !modelica.array<3x2x!modelica.int>
}

// -----

// Dynamic array

// CHECK:       modelica.raw_function @dynamicArray() {
// CHECK-NEXT:      %[[alloca:.*]] = memref.alloca() : memref<memref<3x?xi64>>
// CHECK-NEXT:      %[[fakeArray:.*]] = modelica.alloc : !modelica.array<3x0x!modelica.int>
// CHECK-NEXT:      %[[fakeArray_casted_1:.*]] = builtin.unrealized_conversion_cast %[[fakeArray]] : !modelica.array<3x0x!modelica.int> to memref<3x0xi64>
// CHECK-NEXT:      %[[fakeArray_casted_2:.*]] = memref.cast %[[fakeArray_casted_1]] : memref<3x0xi64> to memref<3x?xi64>
// CHECK-NEXT:      memref.store %[[fakeArray_casted_2]], %[[alloca]][]
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[array:.*]] = memref.load %[[alloca]][]
// CHECK-NEXT:      %[[array_casted:.*]] = builtin.unrealized_conversion_cast %[[array]] : memref<3x?xi64> to !modelica.array<3x?x!modelica.int>
// CHECK-NEXT:      modelica.free %[[array_casted]]
// CHECK-NEXT:      modelica.raw_return
// CHECK-NEXT:  }

modelica.function @dynamicArray : () -> () {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int>
}

// -----

// Load of a dynamic array

// CHECK:   modelica.raw_function @dynamicArrayLoad() {
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

modelica.function @dynamicArrayLoad : () -> () {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int>
    %1 = modelica.member_load %0 : !modelica.member<3x?x!modelica.int>
    %2 = arith.constant 0 : index
    %3 = modelica.load %1[%2, %2] : !modelica.array<3x?x!modelica.int>
}

// -----

// Store of a dynamic array

// CHECK:   modelica.raw_function @dynamicArrayStore() {
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

modelica.function @dynamicArrayStore : () -> () {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int>
    %1 = modelica.alloc : !modelica.array<3x2x!modelica.int>
    modelica.member_store %0, %1 : !modelica.member<3x?x!modelica.int>, !modelica.array<3x2x!modelica.int>
}
