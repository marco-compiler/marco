// RUN: marco-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL: @scalars
// CHECK-SAME: !modelica.bool
// CHECK-SAME: !modelica.int
// CHECK-SAME: !modelica.real
// CHECK-SAME: !modelica.string

func.func @scalars(%arg0: !modelica.bool, %arg1: !modelica.int, %arg2: !modelica.real, %arg3: !modelica.string) {
    return
}

// -----

// CHECK-LABEL: @enumeration
// CHECK-SAME: !modelica.enum<@StateSelect>
// CHECK-SAME: !modelica.enum<@Modelica::@Blocks::@Types::@Init>

func.func @enumeration(%arg0: !modelica.enum<@StateSelect>, %arg1: !modelica.enum<@Modelica::@Blocks::@Types::@Init>) {
    return
}

// -----

// CHECK-LABEL: @object
// CHECK-SAME: !modelica.object<@Modelica::@Electrical::@Analog::@Interfaces::@Pin>

func.func @object(%arg0: !modelica.object<@Modelica::@Electrical::@Analog::@Interfaces::@Pin>) {
    return
}

// -----

// CHECK-LABEL: @arrays
// CHECK-SAME: tensor<3x!modelica.real>
// CHECK-SAME: tensor<?x2x!modelica.int>
// CHECK-SAME: tensor<4x!modelica.object<@Pin>>

func.func @arrays(%arg0: tensor<3x!modelica.real>, %arg1: tensor<?x2x!modelica.int>, %arg2: tensor<4x!modelica.object<@Pin>>) {
    return
}
