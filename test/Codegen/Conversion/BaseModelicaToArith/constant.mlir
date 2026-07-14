// RUN: marco-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// CHECK-LABEL: @Boolean
// CHECK: %[[cst:.*]] = arith.constant true
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i1 to !bmodelica.bool
// CHECK: return %[[result]]

func.func @Boolean() -> !bmodelica.bool {
    %0 = bmodelica.constant #bmodelica<bool true>
    func.return %0 : !bmodelica.bool
}

// -----

// CHECK-LABEL: @Integer
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i64 to !bmodelica.int
// CHECK: return %[[result]]

func.func @Integer() -> !bmodelica.int {
    %0 = bmodelica.constant #bmodelica<int 0>
    func.return %0 : !bmodelica.int
}

// -----

// CHECK-LABEL: @Real
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f64 to !bmodelica.real
// CHECK: return %[[result]]

func.func @Real() -> !bmodelica.real {
    %0 = bmodelica.constant #bmodelica<real 0.0>
    func.return %0 : !bmodelica.real
}

// -----

// CHECK-LABEL: @BooleanTensor
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[false, false, false], [true, true, true]]> : tensor<2x3xi1>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xi1> to tensor<2x3x!bmodelica.bool>
// CHECK: return %[[result]]

func.func @BooleanTensor() -> tensor<2x3x!bmodelica.bool> {
    %0 = bmodelica.constant #bmodelica.dense_bool<[false, false, false, true, true, true]> : tensor<2x3x!bmodelica.bool>
    func.return %0 : tensor<2x3x!bmodelica.bool>
}

// -----

// CHECK-LABEL: @IntegerTensor
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xi64> to tensor<2x3x!bmodelica.int>
// CHECK: return %[[result]]

func.func @IntegerTensor() -> tensor<2x3x!bmodelica.int> {
    %0 = bmodelica.constant #bmodelica.dense_int<[1, 2, 3, 4, 5, 6]> : tensor<2x3x!bmodelica.int>
    func.return %0 : tensor<2x3x!bmodelica.int>
}

// -----

// CHECK-LABEL: @RealTensor
// CHECK: %[[cst:.*]] = arith.constant
// CHECK-SAME{LITERAL}: dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : tensor<2x3xf64> to tensor<2x3x!bmodelica.real>
// CHECK: return %[[result]]

func.func @RealTensor() -> tensor<2x3x!bmodelica.real> {
    %0 = bmodelica.constant #bmodelica.dense_real<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<2x3x!bmodelica.real>
    func.return %0 : tensor<2x3x!bmodelica.real>
}

// -----

// CHECK-LABEL: @mlirIndex
// CHECK: %[[cst:.*]] = arith.constant 0 : index
// CHECK: return %[[cst]]

func.func @mlirIndex() -> index {
    %0 = bmodelica.constant 0 : index
    func.return %0 : index
}

// -----

// CHECK-LABEL: @mlirInteger
// CHECK: %[[cst:.*]] = arith.constant 0 : i64
// CHECK: return %[[cst]]

func.func @mlirInteger() -> i64 {
    %0 = bmodelica.constant 0 : i64
    func.return %0 : i64
}

// -----

// CHECK-LABEL: @mlirFloat
// CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: return %[[cst]]

func.func @mlirFloat() -> f64 {
    %0 = bmodelica.constant 0.0 : f64
    func.return %0 : f64
}

// -----

// CHECK-LABEL: @IntegerTo16BitInteger
// CHECK: %[[cst:.*]] = arith.constant -1 : i16
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i16 to !bmodelica.int
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.int = ["size", 16]>} {
    func.func @IntegerTo16BitInteger() -> !bmodelica.int {
        %0 = bmodelica.constant #bmodelica<int -1>
        func.return %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @IntegerTo32BitInteger
// CHECK: %[[cst:.*]] = arith.constant -1 : i32
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i32 to !bmodelica.int
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.int = ["size", 32]>} {
    func.func @IntegerTo32BitInteger() -> !bmodelica.int {
        %0 = bmodelica.constant #bmodelica<int -1>
        func.return %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @IntegerTo64BitInteger
// CHECK: %[[cst:.*]] = arith.constant -1 : i64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i64 to !bmodelica.int
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.int = ["size", 64]>} {
    func.func @IntegerTo64BitInteger() -> !bmodelica.int {
        %0 = bmodelica.constant #bmodelica<int -1>
        func.return %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @IntegerTo80BitInteger
// CHECK: %[[cst:.*]] = arith.constant -1 : i80
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i80 to !bmodelica.int
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.int = ["size", 80]>} {
    func.func @IntegerTo80BitInteger() -> !bmodelica.int {
        %0 = bmodelica.constant #bmodelica<int -1>
        func.return %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @IntegerTo128BitInteger
// CHECK: %[[cst:.*]] = arith.constant -1 : i128
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : i128 to !bmodelica.int
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.int = ["size", 128]>} {
    func.func @IntegerTo128BitInteger() -> !bmodelica.int {
        %0 = bmodelica.constant #bmodelica<int -1>
        func.return %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @RealTo16BitFloat
// CHECK: %[[cst:.*]] = arith.constant -1.000000e+00 : f16
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f16 to !bmodelica.real
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.real = ["size", 16]>} {
    func.func @RealTo16BitFloat() -> !bmodelica.real {
        %0 = bmodelica.constant #bmodelica<real -1.0>
        func.return %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @RealTo32BitFloat
// CHECK: %[[cst:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f32 to !bmodelica.real
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.real = ["size", 32]>} {
    func.func @RealTo32BitFloat() -> !bmodelica.real {
        %0 = bmodelica.constant #bmodelica<real -1.0>
        func.return %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @RealTo64BitFloat
// CHECK: %[[cst:.*]] = arith.constant -1.000000e+00 : f64
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f64 to !bmodelica.real
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.real = ["size", 64]>} {
    func.func @RealTo64BitFloat() -> !bmodelica.real {
        %0 = bmodelica.constant #bmodelica<real -1.0>
        func.return %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @RealTo80BitFloat
// CHECK: %[[cst:.*]] = arith.constant -1.000000e+00 : f80
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f80 to !bmodelica.real
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.real = ["size", 80]>} {
    func.func @RealTo80BitFloat() -> !bmodelica.real {
        %0 = bmodelica.constant #bmodelica<real -1.0>
        func.return %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @RealTo128BitFloat
// CHECK: %[[cst:.*]] = arith.constant -1.000000e+00 : f128
// CHECK: %[[result:.*]] = builtin.unrealized_conversion_cast %[[cst]] : f128 to !bmodelica.real
// CHECK: return %[[result]]

module attributes {dlti.dl_spec = #dlti.dl_spec<!bmodelica.real = ["size", 128]>} {
    func.func @RealTo128BitFloat() -> !bmodelica.real {
        %0 = bmodelica.constant #bmodelica<real -1.0>
        func.return %0 : !bmodelica.real
    }
}
