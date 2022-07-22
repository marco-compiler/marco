// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// CHECK-LABEL: @boolean
// CHECK-SAME: (%[[arg0:.*]]: !modelica.bool) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.bool to i1
// CHECK-DAG: %[[zero:.*]] = arith.constant false
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @boolean(%arg0 : !modelica.bool) -> !modelica.bool {
    %0 = modelica.not %arg0 : !modelica.bool -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @integer
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @integer(%arg0 : !modelica.int) -> !modelica.bool {
    %0 = modelica.not %arg0 : !modelica.int -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// CHECK-LABEL: @real
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x]], %[[zero]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @real(%arg0 : !modelica.real) -> !modelica.bool {
    %0 = modelica.not %arg0 : !modelica.real -> !modelica.bool
    func.return %0 : !modelica.bool
}
