// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @BooleanToBoolean

func.func @BooleanToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @BooleanToInteger

func.func @BooleanToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 1>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @BooleanToReal

func.func @BooleanToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @BooleanToIndex

func.func @BooleanToIndex() -> (index) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @BooleanToI64

func.func @BooleanToI64() -> (i64) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @BooleanToF64

func.func @BooleanToF64() -> (f64) {
    %x = bmodelica.constant #bmodelica<bool true>
    %result = bmodelica.cast %x: !bmodelica.bool -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 1.000000e+00 : f64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToInteger

func.func @IntegerToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToBoolean

func.func @IntegerToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToReal

func.func @IntegerToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToIndex

func.func @IntegerToIndex() -> (index) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToI64

func.func @IntegerToI64() -> (i64) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IntegerToF64

func.func @IntegerToF64() -> (f64) {
    %x = bmodelica.constant #bmodelica<int 3>
    %result = bmodelica.cast %x: !bmodelica.int -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealToReal

func.func @RealToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealToBoolean

func.func @RealToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant #bmodelica<real 3.0>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealToInteger

func.func @RealToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant #bmodelica<real 3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealToIndex

func.func @RealToIndex() -> (index) {
    %x = bmodelica.constant #bmodelica<real 3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @RealToI64

func.func @RealToI64() -> (i64) {
    %x = bmodelica.constant #bmodelica<real 3.5>
    %result = bmodelica.cast %x : !bmodelica.real -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToIndex

func.func @IndexToIndex() -> (index) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToBoolean

func.func @IndexToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToInteger

func.func @IndexToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToReal

func.func @IndexToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToI64

func.func @IndexToI64() -> (i64) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @IndexToF64

func.func @IndexToF64() -> (f64) {
    %x = bmodelica.constant 3 : index
    %result = bmodelica.cast %x : index -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToI64

func.func @I64ToI64() -> (i64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : i64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToBoolean

func.func @I64ToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToInteger

func.func @I64ToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToReal

func.func @I64ToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToIndex

func.func @I64ToIndex() -> (index) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @I64ToF64

func.func @I64ToF64() -> (f64) {
    %x = bmodelica.constant 3 : i64
    %result = bmodelica.cast %x : i64 -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToF64

func.func @F64ToF64() -> (f64) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.cast %x : f64 -> f64
    return %result : f64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3.000000e+00 : f64
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToBoolean

func.func @F64ToBoolean() -> (!bmodelica.bool) {
    %x = bmodelica.constant 3.0 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.bool
    return %result : !bmodelica.bool
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<bool true>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToInteger

func.func @F64ToInteger() -> (!bmodelica.int) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.int
    return %result : !bmodelica.int
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<int 3>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToReal

func.func @F64ToReal() -> (!bmodelica.real) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> !bmodelica.real
    return %result : !bmodelica.real
    
    // CHECK: %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.500000e+00>
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToIndex

func.func @F64ToIndex() -> (index) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> index
    return %result : index
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : index
    // CHECK: return %[[cst]]
}

// -----

// CHECK-LABEL: @F64ToI64

func.func @F64ToI64() -> (i64) {
    %x = bmodelica.constant 3.5 : f64
    %result = bmodelica.cast %x : f64 -> i64
    return %result : i64
    
    // CHECK: %[[cst:.*]] = bmodelica.constant 3 : i64
    // CHECK: return %[[cst]]
}
