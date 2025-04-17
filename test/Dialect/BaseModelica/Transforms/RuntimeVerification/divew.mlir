// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s

// COM: Integer values

// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x!bmodelica.int>
func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.int>){
    %0 = bmodelica.constant #bmodelica<int 8> : !bmodelica.int

    // CHECK: bmodelica.assert {level = 2 : i64, message = "Model error: element-wise division by zero"} {
    // CHECK-NEXT:   %[[zero:.*]] = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
    // CHECK-NEXT:   %[[neq:.*]] = bmodelica.neq %0, %[[zero]] : (!bmodelica.int, !bmodelica.int) -> !bmodelica.bool
    // CHECK-NEXT:   bmodelica.yield %[[neq]] : !bmodelica.bool
    // CHECK-NEXT: }

    %1 = bmodelica.div_ew %arg0, %0 : (!bmodelica.array<2x!bmodelica.int>, !bmodelica.int) -> !bmodelica.array<2x!bmodelica.int>

    return
}

// -----

// COM: Real values

// RUN: modelica-opt %s --split-input-file --generate-runtime-verification | FileCheck %s


// CHECK-LABEL: @Test
// CHECK-SAME: %{{.*}}: !bmodelica.array<2x!bmodelica.real>
func.func @Test(%arg0: !bmodelica.array<2x!bmodelica.real>){
    %0 = bmodelica.constant #bmodelica<real 8.0> : !bmodelica.real

    // CHECK:        bmodelica.assert {level = 2 : i64, message = "Model error: element-wise division by zero"} {
    // CHECK-NEXT:   %[[const:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
    // CHECK-NEXT:   %[[const_1:.*]] = bmodelica.constant #bmodelica<real 1.000000e-04> : !bmodelica.real
    // CHECK-NEXT:   %[[abs:.*]] = bmodelica.abs %0 : !bmodelica.real -> !bmodelica.real
    // CHECK-NEXT:   %[[gte:.*]] = bmodelica.gte %[[abs]], %[[const_1]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.bool
    // CHECK-NEXT:   bmodelica.yield %[[gte]] : !bmodelica.bool
    // CHECK-NEXT:   }

    %1 = bmodelica.div_ew %arg0, %0 : (!bmodelica.array<2x!bmodelica.real>, !bmodelica.real) -> !bmodelica.array<2x!bmodelica.real>

    return
}