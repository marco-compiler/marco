// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @boolean
// CHECK: modelica.constant #modelica<bool true> : !modelica.bool
// CHECK: modelica.constant #modelica<bool false> : !modelica.bool

func.func @boolean() {
    %0 = modelica.constant #modelica<bool true> : !modelica.bool
    %1 = modelica.constant #modelica<bool false> : !modelica.bool
    return
}

// -----

// CHECK-LABEL: @integer
// CHECK: modelica.constant #modelica<int 42> : !modelica.int
// CHECK: modelica.constant #modelica<int -3> : !modelica.int

func.func @integer() {
    %0 = modelica.constant #modelica<int 42> : !modelica.int
    %1 = modelica.constant #modelica<int -3> : !modelica.int
    return
}

// -----

// CHECK-LABEL: @real
// CHECK: modelica.constant #modelica<real 3.140000e+00> : !modelica.real
// CHECK: modelica.constant #modelica<real -1.000000e-03> : !modelica.real

func.func @real() {
    %0 = modelica.constant #modelica<real 3.14> : !modelica.real
    %1 = modelica.constant #modelica<real -1.0e-3> : !modelica.real
    return
}

// -----

// Unitful literals.

// CHECK-LABEL: @unitful
// CHECK: modelica.constant #modelica<real 9.800000e+00 unit "m/s2"> : !modelica.real
// CHECK: modelica.constant #modelica<int 5 unit "cm"> : !modelica.int

func.func @unitful() {
    %0 = modelica.constant #modelica<real 9.8 unit "m/s2"> : !modelica.real
    %1 = modelica.constant #modelica<int 5 unit "cm"> : !modelica.int
    return
}

// -----

// CHECK-LABEL: @string
// CHECK: modelica.constant #modelica<string "Ohm"> : !modelica.string
// CHECK: modelica.constant #modelica<string "a\22b"> : !modelica.string

func.func @string() {
    %0 = modelica.constant #modelica<string "Ohm"> : !modelica.string
    %1 = modelica.constant #modelica<string "a\"b"> : !modelica.string
    return
}

// -----

// CHECK-LABEL: @enumeration
// CHECK: modelica.constant #modelica<enum "always" = 5> : !modelica.enum<@StateSelect>

func.func @enumeration() {
    %0 = modelica.constant #modelica<enum "always" = 5> : !modelica.enum<@StateSelect>
    return
}

// -----

// CHECK-LABEL: @arrays
// CHECK: modelica.constant #modelica<array [1, 2, 3, 4]> : tensor<2x2x!modelica.int>
// CHECK: modelica.constant #modelica<array [1.500000e+00, 2.000000e+00 unit "m"]> : tensor<2x!modelica.real>
// CHECK: modelica.constant #modelica<array [true, false]> : tensor<2x!modelica.bool>
// CHECK: modelica.constant #modelica<array ["a", "b"]> : tensor<2x!modelica.string>
// CHECK: modelica.constant #modelica<array ["x" = 1, "y" = 2]> : tensor<2x!modelica.enum<@E>>

func.func @arrays() {
    %0 = modelica.constant #modelica<array [1, 2, 3, 4]> : tensor<2x2x!modelica.int>
    %1 = modelica.constant #modelica<array [1.5, 2.0 unit "m"]> : tensor<2x!modelica.real>
    %2 = modelica.constant #modelica<array [true, false]> : tensor<2x!modelica.bool>
    %3 = modelica.constant #modelica<array ["a", "b"]> : tensor<2x!modelica.string>
    %4 = modelica.constant #modelica<array ["x" = 1, "y" = 2]> : tensor<2x!modelica.enum<@E>>
    return
}
