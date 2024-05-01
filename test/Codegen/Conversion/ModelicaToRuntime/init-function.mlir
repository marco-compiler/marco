// RUN: modelica-opt %s --split-input-file --convert-modelica-to-runtime | FileCheck %s

// CHECK-DAG: bmodelica.global_variable @[[global_var:.*]] : !bmodelica.array<!bmodelica.int>

// CHECK:       runtime.init_function {
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[x:.*]] = bmodelica.global_variable_get @[[global_var]]
// CHECK:           bmodelica.store %[[x]][], %[[zero]]
// CHECK-NEXT:      runtime.yield
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>
    }
}

// -----

// CHECK-DAG: bmodelica.global_variable @[[global_var:.*]] : !bmodelica.array<3x!bmodelica.int>

// CHECK:       runtime.init_function {
// CHECK-DAG:       %[[zero:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[x:.*]] = bmodelica.global_variable_get @[[global_var]]
// CHECK:           bmodelica.array_fill %[[x]], %[[zero]]
// CHECK-NEXT:      runtime.yield
// CHECK-NEXT:  }

module {
    bmodelica.model @Test {
        bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>
    }
}
