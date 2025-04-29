// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf | FileCheck %s

// CHECK-LABEL: @scalarVariableGet
// CHECK-SAME:  () -> i64
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<i64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = bmodelica.raw_variable_get %[[variable]]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      cf.br ^{{.*}}
// CHECK:       }

bmodelica.function @scalarVariableGet {
    bmodelica.variable @x : !bmodelica.variable<i64, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : i64
        bmodelica.print %0 : i64
    }
}

// -----

// CHECK-LABEL: @scalarVariableSet
// CHECK-SAME:  () -> i64
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<i64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = arith.constant 0 : i64
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[value]]
// CHECK-NEXT:      cf.br ^{{.*}}
// CHECK:       }

bmodelica.function @scalarVariableSet {
    bmodelica.variable @x : !bmodelica.variable<i64, output>

    bmodelica.algorithm {
        %0 = arith.constant 0 : i64
        bmodelica.variable_set @x, %0 : i64
    }
}

// -----

// CHECK-LABEL: @staticArrayGet
// CHECK-SAME:  () -> tensor<3x2xi64>
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<3x2xi64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = bmodelica.raw_variable_get %[[variable]]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      cf.br ^{{.*}}
// CHECK:       }

bmodelica.function @staticArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x2xi64, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x2xi64>
        bmodelica.print %0 : tensor<3x2xi64>
    }
}

// -----

// CHECK-LABEL: @staticArraySet
// CHECK-SAME:  () -> tensor<3x2xi64>
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<3x2xi64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = tensor.empty() : tensor<3x2xi64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[value]]
// CHECK-NEXT:      cf.br ^{{.*}}
// CHECK:       }

bmodelica.function @staticArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x2xi64, output>

    bmodelica.algorithm {
        %0 = tensor.empty() : tensor<3x2xi64>
        bmodelica.variable_set @x, %0 : tensor<3x2xi64>
    }
}

// -----

// CHECK-LABEL: @dynamicArrayGet
// CHECK-SAME:  () -> tensor<3x?xi64>
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<3x?xi64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = bmodelica.raw_variable_get %[[variable]]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      cf.br ^{{.*}}
// CHECK:       }

bmodelica.function @dynamicArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x?xi64, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x?xi64>
        bmodelica.print %0 : tensor<3x?xi64>
    }
}

// -----

// CHECK-LABEL: @dynamicArraySet
// CHECK-SAME:  () -> tensor<3x?xi64>
// CHECK-NEXT:      cf.br ^[[bb1:.*]]
// CHECK-NEXT:  ^[[bb1]]:
// CHECK-NEXT:      %[[variable:.*]] = bmodelica.raw_variable {name = "x", output} : tensor<3x?xi64>
// CHECK-NEXT:      cf.br ^[[bb2:.*]]
// CHECK-NEXT:  ^[[bb2]]:
// CHECK-NEXT:      %[[value:.*]] = tensor.empty() : tensor<3x2xi64>
// CHECK-NEXT:      %[[cast:.*]] = bmodelica.cast %[[value]] : tensor<3x2xi64> -> tensor<3x?xi64>
// CHECK-NEXT:      bmodelica.raw_variable_set %[[variable]], %[[cast]]
// CHECK:       }

bmodelica.function @dynamicArraySet {
    bmodelica.variable @x : !bmodelica.variable<3x?xi64, output>

    bmodelica.algorithm {
        %0 = tensor.empty() : tensor<3x2xi64>
        bmodelica.variable_set @x, %0 : tensor<3x2xi64>
    }
}
