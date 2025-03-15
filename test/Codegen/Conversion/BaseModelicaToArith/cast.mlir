// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// CHECK-LABEL: @BooleanToBoolean
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: return %[[arg0]]

func.func @BooleanToBoolean(%arg0 : !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @BooleanToInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[ext]]
// CHECK: return %[[result]]

func.func @BooleanToInteger(%arg0 : !bmodelica.bool) -> !bmodelica.int {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @BooleanToReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i64
// CHECK: %[[fp:.*]] = arith.sitofp %[[ext]] : i64 to f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[fp]]
// CHECK: return %[[result]]

func.func @BooleanToReal(%arg0 : !bmodelica.bool) -> !bmodelica.real {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @BooleanToI1
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: return %[[result]]

func.func @BooleanToI1(%arg0 : !bmodelica.bool) -> i1 {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> i1
    func.return %0 : i1
}

// -----

// CHECK-LABEL: @BooleanToI32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i32
// CHECK: return %[[ext]]

func.func @BooleanToI32(%arg0 : !bmodelica.bool) -> i32 {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> i32
    func.return %0 : i32
}

// -----

// CHECK-LABEL: @BooleanToI64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i64
// CHECK: return %[[ext]]

func.func @BooleanToI64(%arg0 : !bmodelica.bool) -> i64 {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @BooleanToF32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i32
// CHECK: %[[fp:.*]] = arith.sitofp %[[ext]] : i32 to f32
// CHECK: return %[[fp]]

func.func @BooleanToF32(%arg0 : !bmodelica.bool) -> f32 {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> f32
    func.return %0 : f32
}

// -----

// CHECK-LABEL: @BooleanToF64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK: %[[ext:.*]] = arith.extui %[[casted]] : i1 to i64
// CHECK: %[[fp:.*]] = arith.sitofp %[[ext]] : i64 to f64
// CHECK: return %[[fp]]

func.func @BooleanToF64(%arg0 : !bmodelica.bool) -> f64 {
    %0 = bmodelica.cast %arg0: !bmodelica.bool -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @IntegerToBoolean
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[trunc:.*]] = arith.trunci %[[casted]] : i64 to i1
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[trunc]]
// CHECK: return %[[result]]

func.func @IntegerToBoolean(%arg0 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @IntegerToInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: return %[[arg0]]

func.func @IntegerToInteger(%arg0 : !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @IntegerToReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[fp:.*]] = arith.sitofp %[[casted]] : i64 to f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[fp]]
// CHECK: return %[[result]]

func.func @IntegerToReal(%arg0 : !bmodelica.int) -> !bmodelica.real {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @IntegerToI1
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[trunc:.*]] = arith.trunci %[[casted]] : i64 to i1
// CHECK: return %[[trunc]]

func.func @IntegerToI1(%arg0 : !bmodelica.int) -> i1 {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> i1
    func.return %0 : i1
}

// -----

// CHECK-LABEL: @IntegerToI32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[trunc:.*]] = arith.trunci %[[casted]] : i64 to i32
// CHECK: return %[[trunc]]

func.func @IntegerToI32(%arg0 : !bmodelica.int) -> i32 {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> i32
    func.return %0 : i32
}

// -----

// CHECK-LABEL: @IntegerToI64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: return %[[casted]]

func.func @IntegerToI64(%arg0 : !bmodelica.int) -> i64 {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @IntegerToF32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[ext:.*]] = arith.trunci %[[casted]] : i64 to i32
// CHECK: %[[fp:.*]] = arith.sitofp %[[ext]] : i32 to f32
// CHECK: return %[[fp]]

func.func @IntegerToF32(%arg0 : !bmodelica.int) -> f32 {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> f32
    func.return %0 : f32
}

// -----

// CHECK-LABEL: @IntegerToF64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK: %[[fp:.*]] = arith.sitofp %[[casted]] : i64 to f64
// CHECK: return %[[fp]]

func.func @IntegerToF64(%arg0 : !bmodelica.int) -> f64 {
    %0 = bmodelica.cast %arg0: !bmodelica.int -> f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @RealToBoolean
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[si:.*]] = arith.fptosi %[[casted]] : f64 to i1
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[si]]
// CHECK: return %[[result]]

func.func @RealToBoolean(%arg0 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @RealToInteger
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[si:.*]] = arith.fptosi %[[casted]] : f64 to i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[si]]
// CHECK: return %[[result]]

func.func @RealToInteger(%arg0 : !bmodelica.real) -> !bmodelica.int {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @RealToReal
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: return %[[arg0]]

func.func @RealToReal(%arg0 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @RealToI1
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[si:.*]] = arith.fptosi %[[casted]] : f64 to i1
// CHECK: return %[[si]]

func.func @RealToI1(%arg0 : !bmodelica.real) -> i1 {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> i1
    func.return %0 : i1
}

// -----

// CHECK-LABEL: @RealToI32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[si:.*]] = arith.fptosi %[[casted]] : f64 to i32
// CHECK: return %[[si]]

func.func @RealToI32(%arg0 : !bmodelica.real) -> i32 {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> i32
    func.return %0 : i32
}

// -----

// CHECK-LABEL: @RealToI64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[si:.*]] = arith.fptosi %[[casted]] : f64 to i64
// CHECK: return %[[si]]

func.func @RealToI64(%arg0 : !bmodelica.real) -> i64 {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @RealToF32
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: %[[trunc:.*]] = arith.truncf %[[casted]] : f64 to f32
// CHECK: return %[[fp]]

func.func @RealToF32(%arg0 : !bmodelica.real) -> f32 {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> f32
    func.return %0 : f32
}

// -----

// CHECK-LABEL: @RealToF64
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real)
// CHECK: %[[casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK: return %[[casted]]

func.func @RealToF64(%arg0 : !bmodelica.real) -> f64 {
    %0 = bmodelica.cast %arg0: !bmodelica.real -> f64
    func.return %0 : f64
}
