// RUN: modelica-opt %s --split-input-file --function-default-values-conversion | FileCheck %s

// CHECK-LABEL: @caller
// CHECK: %[[default:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK: bmodelica.call @scalarDefaultValue(%[[default]]) : (!bmodelica.int) -> ()

bmodelica.function @scalarDefaultValue {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.default @x {
        %0 = bmodelica.constant #bmodelica<int 0>
        bmodelica.yield %0 : !bmodelica.int
    }

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %0 : !bmodelica.int
    }
}

func.func @caller() {
    bmodelica.call @scalarDefaultValue() : () -> ()
    func.return
}
