// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @scalarOutputAsArgument
// CHECK:   bmodelica.algorithm {
// CHECK-DAG:   %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:   %[[alloc:.*]] = bmodelica.alloc : <!bmodelica.int>
// CHECK:       bmodelica.store %[[alloc]][], %[[x]]
// CHECK:       bmodelica.external_call @foo(%[[alloc]]) : (!bmodelica.array<!bmodelica.int>) -> ()
// CHECK:       %[[result:.*]] = bmodelica.load %[[alloc]][]
// CHECK:       bmodelica.variable_set @x, %[[result]]
// CHECK:   }

function scalarOutputAsArgument
    output Integer x;
external "C" foo(x);
end scalarOutputAsArgument;
