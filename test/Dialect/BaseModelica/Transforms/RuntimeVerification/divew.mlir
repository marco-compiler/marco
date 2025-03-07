// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// Integer values

// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x!bmodelica.int>>

    %0 = bmodelica.constant #bmodelica<int 8> : !bmodelica.int
    %1 = bmodelica.variable_get @arg : !bmodelica.array<2x!bmodelica.int>

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: element-wise division by zero"} {
    // CHECK-NEXT:   %[[neq:.*]] = bmodelica.neq %0, %2 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:   bmodelica.yield %[[neq]] : !bmodelica.bool
    // CHECK-NEXT: }

    %2 = bmodelica.div_ew %1, %0 : (!bmodelica.array<2x!bmodelica.int>, !bmodelica.int) -> !bmodelica.array<2x!bmodelica.int>

}

// -----

// Real values

// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s


// CHECK-LABEL: @Test
bmodelica.model @Test {
    bmodelica.variable @arg : !bmodelica.variable<!bmodelica.array<2x!bmodelica.real>>

    %0 = bmodelica.constant #bmodelica<real 8.0> : !bmodelica.real
    %1 = bmodelica.variable_get @arg : !bmodelica.array<2x!bmodelica.real>

    // CHECK:        %[[const:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:   bmodelica.assert {level = 2 : i64, message = "Model error: element-wise division by zero"} {
    // CHECK-NEXT:   %[[const_1:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:   %[[abs:.*]] = bmodelica.abs %0 : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:   %[[gte:.*]] = bmodelica.gte %[[abs]], %[[const_1]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:   bmodelica.yield %[[gte]] : !bmodelica.bool
    // CHECK-NEXT:   }

    %2 = bmodelica.div_ew %1, %0 : (!bmodelica.array<2x!bmodelica.real>, !bmodelica.real) -> !bmodelica.array<2x!bmodelica.real>

}