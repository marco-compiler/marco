// RUN: modelica-opt %s --split-input-file --pass-pipeline="builtin.module(convert-modelica-to-cf{output-arrays-promotion=false})" | FileCheck %s

// Scalar variable

// CHECK:       modelica.raw_function @foo() -> !modelica.int {
// CHECK:           %[[y:.*]] = modelica.alloca : !modelica.array<!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      %[[result:.*]] = modelica.load %[[y]][] : !modelica.array<!modelica.int>
// CHECK-NEXT:      modelica.raw_return %[[result]]
// CHECK-NEXT:  }

modelica.function @foo : () -> (!modelica.int) {
    %0 = modelica.member_create @y : !modelica.member<!modelica.int, output>
}

// -----

// Static array

// CHECK:       modelica.raw_function @foo() -> !modelica.array<3x2x!modelica.int> {
// CHECK:           %[[y:.*]] = modelica.alloc : !modelica.array<3x2x!modelica.int>
// CHECK-NEXT:      cf.br ^[[out:.*]]
// CHECK-NEXT:  ^[[out]]:  // pred: ^bb0
// CHECK-NEXT:      modelica.raw_return %[[y]]
// CHECK-NEXT:  }

modelica.function @foo : () -> (!modelica.array<3x2x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x2x!modelica.int, output>
}

// -----

// Dynamic array

// CHECK:       modelica.raw_function @foo() -> !modelica.array<3x?x!modelica.int> {
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

modelica.function @foo : () -> (!modelica.array<3x?x!modelica.int>) {
    %0 = modelica.member_create @y : !modelica.member<3x?x!modelica.int, output>
}
