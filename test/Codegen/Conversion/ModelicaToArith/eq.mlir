// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[y]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.eq %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.eq %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[x_f64:.*]] = arith.sitofp %[[x]] : i64 to f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x_f64]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.eq %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[y_f64:.*]] = arith.sitofp %[[y]] : i64 to f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x]], %[[y_f64]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.eq %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> i1
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[arg1]] : index
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : index, %arg1 : index) -> i1 {
    %0 = modelica.eq %arg0, %arg1 : (index, index) -> i1
    func.return %0 : i1
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i1
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[arg1]] : i64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : i64, %arg1 : i64) -> i1 {
    %0 = modelica.eq %arg0, %arg1 : (i64, i64) -> i1
    func.return %0 : i1
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> i1
// CHECK: %[[result:.*]] = arith.cmpf oeq, %[[arg0]], %[[arg1]] : f64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : f64, %arg1 : f64) -> i1 {
    %0 = modelica.eq %arg0, %arg1 : (f64, f64) -> i1
    func.return %0 : i1
}
