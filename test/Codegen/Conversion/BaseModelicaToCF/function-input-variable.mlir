// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-cf --canonicalize --cse | FileCheck %s

// CHECK-LABEL: @scalarVariable
// CHECK-SAME:  (%{{.*}}: !bmodelica.int)
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @scalarVariable {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>
}

// -----

// CHECK-LABEL: @scalarVariableGet
// CHECK-SAME:  (%[[x:.*]]: !bmodelica.int)
// CHECK-NEXT:      bmodelica.print %[[x]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @scalarVariableGet {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.int
        bmodelica.print %0 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @staticArray
// CHECK-SAME:  (%{{.*}}: tensor<3x2x!bmodelica.int>)
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @staticArray {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, input>
}

// -----

// CHECK-LABEL: @staticArrayGet
// CHECK-SAME:  (%[[x:.*]]: tensor<3x2x!bmodelica.int>)
// CHECK:           %[[value:.*]] = bmodelica.tensor_extract %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @staticArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x2x!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x2x!bmodelica.int>
        %1 = arith.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1, %1] : tensor<3x2x!bmodelica.int>
        bmodelica.print %2 : !bmodelica.int
    }
}

// -----

// CHECK-LABEL: @dynamicArray
// CHECK-SAME:  (%{{.*}}: tensor<3x?x!bmodelica.int>)
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @dynamicArray {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, input>
}

// -----

// CHECK-LABEL: @dynamicArrayGet
// CHECK-SAME:  (%[[x:.*]]: tensor<3x?x!bmodelica.int>)
// CHECK-NEXT:      %[[index:.*]]= arith.constant 0 : index
// CHECK-NEXT:      %[[value:.*]] = bmodelica.tensor_extract %[[x]][%{{.*}}, %{{.*}}]
// CHECK-NEXT:      bmodelica.print %[[value]]
// CHECK-NEXT:      bmodelica.raw_return
// CHECK-NEXT:  }

bmodelica.function @dynamicArrayGet {
    bmodelica.variable @x : !bmodelica.variable<3x?x!bmodelica.int, input>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : tensor<3x?x!bmodelica.int>
        %1 = arith.constant 0 : index
        %2 = bmodelica.tensor_extract %0[%1, %1] : tensor<3x?x!bmodelica.int>
        bmodelica.print %2 : !bmodelica.int
    }
}
