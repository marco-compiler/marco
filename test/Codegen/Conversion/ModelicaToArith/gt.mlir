// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @integers
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[cmp:.*]] = arith.cmpi sgt, %[[x]], %[[y]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @integers(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.gt %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @reals
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @reals(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.gt %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @integerReal
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[arg0_casted:.*]] = modelica.cast %[[arg0]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0_casted]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @integerReal(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.gt %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @realInteger
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[arg1_casted:.*]] = modelica.cast %[[arg1]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1_casted]] : !modelica.real to f64
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf ogt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @realInteger(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.gt %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}
