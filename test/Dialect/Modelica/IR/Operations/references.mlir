// RUN: marco-opt %s --split-input-file | FileCheck %s

// Component references.

// CHECK-LABEL: @references
// CHECK: modelica.ref @x : tensor<3x!modelica.real>
// CHECK: modelica.ref @"'quoted name'" : !modelica.real
// CHECK: modelica.member %{{.*}}, @v : (!modelica.object<@Pin>) -> !modelica.real
// CHECK: modelica.member %{{.*}}, @p : (tensor<10x!modelica.object<@Resistor>>) -> tensor<10x!modelica.object<@Pin>>
// CHECK: modelica.member %{{.*}}, @x : (tensor<2x!modelica.object<@R>>) -> tensor<2x3x!modelica.real>
// CHECK: modelica.ref @Modelica::@Constants::@pi : !modelica.real
// CHECK: modelica.ref @"."::@Modelica::@Constants::@e : !modelica.real

func.func @references(%pin: !modelica.object<@Pin>, %resistors: tensor<10x!modelica.object<@Resistor>>, %records: tensor<2x!modelica.object<@R>>) {
    %0 = modelica.ref @x : tensor<3x!modelica.real>
    %1 = modelica.ref @"'quoted name'" : !modelica.real
    %2 = modelica.member %pin, @v : (!modelica.object<@Pin>) -> !modelica.real
    // Slice operations.
    %3 = modelica.member %resistors, @p : (tensor<10x!modelica.object<@Resistor>>) -> tensor<10x!modelica.object<@Pin>>
    %4 = modelica.member %records, @x : (tensor<2x!modelica.object<@R>>) -> tensor<2x3x!modelica.real>
    %5 = modelica.ref @Modelica::@Constants::@pi : !modelica.real
    %6 = modelica.ref @"."::@Modelica::@Constants::@e : !modelica.real
    return
}

// -----

// Indexing.

// CHECK-LABEL: @subscripts
// CHECK: modelica.subscript %{{.*}}[%{{.*}}, %{{.*}}] : (tensor<3x4x!modelica.real>, !modelica.int, !modelica.int) -> !modelica.real
// CHECK: modelica.subscript %{{.*}}[:, %{{.*}}] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<3x!modelica.real>
// CHECK: modelica.subscript %{{.*}}[%{{.*}}] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<4x!modelica.real>
// CHECK: modelica.subscript %{{.*}}[%{{.*}}, :] : (tensor<3x4x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x4x!modelica.real>
// CHECK: modelica.subscript %{{.*}}[%{{.*}}] : (tensor<3x!modelica.real>, !modelica.enum<@E>) -> !modelica.real
// CHECK: modelica.subscript %{{.*}}[%{{.*}}] : (tensor<2x!modelica.real>, !modelica.bool) -> !modelica.real

func.func @subscripts(%a: tensor<3x4x!modelica.real>, %i: !modelica.int, %v: tensor<2x!modelica.int>, %w: tensor<3x!modelica.real>, %e: !modelica.enum<@E>, %b: tensor<2x!modelica.real>, %c: !modelica.bool) {
    %0 = modelica.subscript %a[%i, %i] : (tensor<3x4x!modelica.real>, !modelica.int, !modelica.int) -> !modelica.real
    %1 = modelica.subscript %a[:, %i] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<3x!modelica.real>
    %2 = modelica.subscript %a[%i] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<4x!modelica.real>
    %3 = modelica.subscript %a[%v, :] : (tensor<3x4x!modelica.real>, tensor<2x!modelica.int>) -> tensor<2x4x!modelica.real>
    // Enumeration and Boolean indices.
    %4 = modelica.subscript %w[%e] : (tensor<3x!modelica.real>, !modelica.enum<@E>) -> !modelica.real
    %5 = modelica.subscript %b[%c] : (tensor<2x!modelica.real>, !modelica.bool) -> !modelica.real
    return
}

// -----

// Function calls.

// CHECK-LABEL: @calls
// CHECK: modelica.call @Modelica::@Math::@sin(%{{.*}}) : (!modelica.real) -> !modelica.real
// CHECK: modelica.call @f(%{{.*}}, @y = %{{.*}}, @"'z'" = %{{.*}}) : (!modelica.real, !modelica.int, !modelica.bool) -> (!modelica.real, !modelica.real)
// CHECK: modelica.call @R(@x = %{{.*}}) : (!modelica.real) -> !modelica.object<@R>
// CHECK: modelica.call @g() : () -> ()

func.func @calls(%x: !modelica.real, %y: !modelica.int, %z: !modelica.bool) {
    %0 = modelica.call @Modelica::@Math::@sin(%x) : (!modelica.real) -> !modelica.real
    %1:2 = modelica.call @f(%x, @y = %y, @"'z'" = %z) : (!modelica.real, !modelica.int, !modelica.bool) -> (!modelica.real, !modelica.real)
    // Record constructor.
    %2 = modelica.call @R(@x = %x) : (!modelica.real) -> !modelica.object<@R>
    modelica.call @g() : () -> ()
    return
}

// -----

// Operators on connectors and special operators.

// CHECK-LABEL: @connectorOperators
// CHECK: modelica.in_stream %{{.*}} : (!modelica.real) -> !modelica.real
// CHECK: modelica.actual_stream %{{.*}} : (!modelica.real) -> !modelica.real
// CHECK: modelica.cardinality %{{.*}} : (!modelica.object<@Pin>) -> !modelica.int
// CHECK: modelica.get_instance_name : !modelica.string
// CHECK: modelica.spatial_distribution %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!modelica.real, !modelica.real, !modelica.real, !modelica.bool) -> (!modelica.real, !modelica.real)
// CHECK: modelica.spatial_distribution %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} initialPoints %{{.*}} initialValues %{{.*}} :
// CHECK: modelica.connections_is_root %{{.*}} : (!modelica.object<@Orientation>) -> !modelica.bool
// CHECK: modelica.connections_rooted %{{.*}} : (!modelica.object<@Orientation>) -> !modelica.bool
// CHECK: modelica.connections_unique_root_indices %{{.*}}, %{{.*}}, %{{.*}} : (tensor<3x!modelica.object<@Node>>, tensor<?x!modelica.object<@Node>>, !modelica.string) -> tensor<3x!modelica.int>

func.func @connectorOperators(%x: !modelica.real, %v: !modelica.bool, %points: tensor<2x!modelica.real>, %message: !modelica.string) {
    %port = modelica.ref @port : !modelica.object<@FluidPort>
    %h = modelica.member %port, @h_outflow : (!modelica.object<@FluidPort>) -> !modelica.real
    %0 = modelica.in_stream %h : (!modelica.real) -> !modelica.real
    %1 = modelica.actual_stream %h : (!modelica.real) -> !modelica.real
    %pin = modelica.ref @p : !modelica.object<@Pin>
    %2 = modelica.cardinality %pin : (!modelica.object<@Pin>) -> !modelica.int
    %3 = modelica.get_instance_name : !modelica.string
    %4:2 = modelica.spatial_distribution %x, %x, %x, %v : (!modelica.real, !modelica.real, !modelica.real, !modelica.bool) -> (!modelica.real, !modelica.real)
    %5:2 = modelica.spatial_distribution %x, %x, %x, %v initialPoints %points initialValues %points : (!modelica.real, !modelica.real, !modelica.real, !modelica.bool, tensor<2x!modelica.real>, tensor<2x!modelica.real>) -> (!modelica.real, !modelica.real)
    %frame = modelica.ref @frame : !modelica.object<@Frame>
    %R = modelica.member %frame, @R : (!modelica.object<@Frame>) -> !modelica.object<@Orientation>
    %6 = modelica.connections_is_root %R : (!modelica.object<@Orientation>) -> !modelica.bool
    %7 = modelica.connections_rooted %R : (!modelica.object<@Orientation>) -> !modelica.bool
    %roots = modelica.ref @roots : tensor<3x!modelica.object<@Node>>
    %nodes = modelica.ref @nodes : tensor<?x!modelica.object<@Node>>
    %8 = modelica.connections_unique_root_indices %roots, %nodes, %message : (tensor<3x!modelica.object<@Node>>, tensor<?x!modelica.object<@Node>>, !modelica.string) -> tensor<3x!modelica.int>
    return
}
