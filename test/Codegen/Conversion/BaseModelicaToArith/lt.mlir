// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[cmp:.*]] = arith.cmpi slt, %[[x]], %[[y]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.lt %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.lt %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[x_f64:.*]] = arith.sitofp %[[x]] : i64 to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x_f64]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.lt %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[y_f64:.*]] = arith.sitofp %[[y]] : i64 to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x]], %[[y_f64]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.lt %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> i1
// CHECK: %[[result:.*]] = arith.cmpi slt, %[[arg0]], %[[arg1]] : index
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : index, %arg1 : index) -> i1 {
    %0 = bmodelica.lt %arg0, %arg1 : (index, index) -> i1
    func.return %0 : i1
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i1
// CHECK: %[[result:.*]] = arith.cmpi slt, %[[arg0]], %[[arg1]] : i64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : i64, %arg1 : i64) -> i1 {
    %0 = bmodelica.lt %arg0, %arg1 : (i64, i64) -> i1
    func.return %0 : i1
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> i1
// CHECK: %[[result:.*]] = arith.cmpf olt, %[[arg0]], %[[arg1]] : f64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : f64, %arg1 : f64) -> i1 {
    %0 = bmodelica.lt %arg0, %arg1 : (f64, f64) -> i1
    func.return %0 : i1
}
