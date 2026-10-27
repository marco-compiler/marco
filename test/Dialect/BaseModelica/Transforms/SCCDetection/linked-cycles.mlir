// RUN: marco-opt %s --split-input-file --mlir-disable-threading --detect-scc --canonicalize-model-for-debug | FileCheck %s
// RUN: marco-opt %s --split-input-file --mlir-disable-threading --detect-scc --canonicalize-model-for-debug | FileCheck %s --check-prefix=COUNT

// COM: Two cycles ({x, y} and {z, w}) that are linked by a third cycle
// COM: (x -> z -> w -> y -> x) passing through both of them. All the
// COM: equations belong to the same SCC.

// CHECK-LABEL: @LinkedScalarCycles

bmodelica.model @LinkedScalarCycles {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @w : !bmodelica.variable<!bmodelica.real>

    // COM: x = y + z
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable.get @x : !bmodelica.real
        %1 = bmodelica.variable.get @y : !bmodelica.real
        %2 = bmodelica.variable.get @z : !bmodelica.real
        %3 = bmodelica.add %1, %2 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: y = 1 - x
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable.get @y : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.variable.get @x : !bmodelica.real
        %3 = bmodelica.sub %1, %2 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: z = 2 - w
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable.get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 2.0>
        %2 = bmodelica.variable.get @w : !bmodelica.real
        %3 = bmodelica.sub %1, %2 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}

    // COM: w = z + y
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable.get @w : !bmodelica.real
        %1 = bmodelica.variable.get @z : !bmodelica.real
        %2 = bmodelica.variable.get @y : !bmodelica.real
        %3 = bmodelica.add %1, %2 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, match = @x
        bmodelica.equation_instance %t1, match = @y
        bmodelica.equation_instance %t2, match = @z
        bmodelica.equation_instance %t3, match = @w
    }

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t0]], match = @x
    // CHECK-DAG: bmodelica.equation_instance %[[t1]], match = @y
    // CHECK-DAG: bmodelica.equation_instance %[[t2]], match = @z
    // CHECK-DAG: bmodelica.equation_instance %[[t3]], match = @w

    // COUNT-LABEL: @LinkedScalarCycles
    // COUNT-COUNT-1: bmodelica.scc
    // COUNT-NOT: bmodelica.scc

    // CHECK-NOT: bmodelica.equation_instance
}

// -----

// COM: For each index i, the array equations form two cycles ({x[i], y[i]}
// COM: and {z[i], w[i]}). For i = 0, the scalar equations link the two cycles
// COM: through a third one (x[0] -> z[0] -> w[0] -> y[0] -> x[0]), so the four
// COM: equations writing to index 0 belong to the same SCC. The cycles for
// COM: i = 1 must instead remain separate.

// CHECK-LABEL: @LinkedArrayCycles

bmodelica.model @LinkedArrayCycles {
    bmodelica.variable @x : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<2x!bmodelica.real>
    bmodelica.variable @w : !bmodelica.variable<2x!bmodelica.real>

    // COM: x[i] = 1 - y[i]
    %t0 = bmodelica.equation_template inductions = [%i0] attributes {id = "t0"} {
        %0 = bmodelica.variable.get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 1.0>
        %3 = bmodelica.variable.get @y : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor.extract %3[%i0] : tensor<2x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t0"}

    // COM: y[i] = 2 - x[i]
    %t1 = bmodelica.equation_template inductions = [%i0] attributes {id = "t1"} {
        %0 = bmodelica.variable.get @y : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 2.0>
        %3 = bmodelica.variable.get @x : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor.extract %3[%i0] : tensor<2x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t1"}

    // COM: z[i] = 3 - w[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable.get @z : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 3.0>
        %3 = bmodelica.variable.get @w : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor.extract %3[%i0] : tensor<2x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    // COM: w[i] = 4 - z[i]
    %t3 = bmodelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %0 = bmodelica.variable.get @w : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 4.0>
        %3 = bmodelica.variable.get @z : tensor<2x!bmodelica.real>
        %4 = bmodelica.tensor.extract %3[%i0] : tensor<2x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}

    // COM: x[0] = y[0] + z[0]
    %t4 = bmodelica.equation_template inductions = [] attributes {id = "t4"} {
        %c0 = bmodelica.constant 0 : index
        %0 = bmodelica.variable.get @x : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%c0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.variable.get @y : tensor<2x!bmodelica.real>
        %3 = bmodelica.tensor.extract %2[%c0] : tensor<2x!bmodelica.real>
        %4 = bmodelica.variable.get @z : tensor<2x!bmodelica.real>
        %5 = bmodelica.tensor.extract %4[%c0] : tensor<2x!bmodelica.real>
        %6 = bmodelica.add %3, %5 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t4"}

    // COM: w[0] = z[0] + y[0]
    %t5 = bmodelica.equation_template inductions = [] attributes {id = "t5"} {
        %c0 = bmodelica.constant 0 : index
        %0 = bmodelica.variable.get @w : tensor<2x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%c0] : tensor<2x!bmodelica.real>
        %2 = bmodelica.variable.get @z : tensor<2x!bmodelica.real>
        %3 = bmodelica.tensor.extract %2[%c0] : tensor<2x!bmodelica.real>
        %4 = bmodelica.variable.get @y : tensor<2x!bmodelica.real>
        %5 = bmodelica.tensor.extract %4[%c0] : tensor<2x!bmodelica.real>
        %6 = bmodelica.add %3, %5 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t5:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t5"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t0, indices = {[1,1]}, match = <@x, {[1,1]}>
        bmodelica.equation_instance %t1, indices = {[0,1]}, match = <@y, {[0,1]}>
        bmodelica.equation_instance %t2, indices = {[0,1]}, match = <@z, {[0,1]}>
        bmodelica.equation_instance %t3, indices = {[1,1]}, match = <@w, {[1,1]}>
        bmodelica.equation_instance %t4, match = <@x, {[0,0]}>
        bmodelica.equation_instance %t5, match = <@w, {[0,0]}>
    }

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t0]], indices = {[1,1]}, match = <@x, {[1,1]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t1]], indices = {[1,1]}, match = <@y, {[1,1]}>

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t1]], indices = {[0,0]}, match = <@y, {[0,0]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t2]], indices = {[0,0]}, match = <@z, {[0,0]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t4]], match = <@x, {[0,0]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t5]], match = <@w, {[0,0]}>

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t2]], indices = {[1,1]}, match = <@z, {[1,1]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t3]], indices = {[1,1]}, match = <@w, {[1,1]}>

    // CHECK-NOT: bmodelica.equation_instance

    // COUNT-LABEL: @LinkedArrayCycles
    // COUNT-COUNT-3: bmodelica.scc
    // COUNT-NOT: bmodelica.scc
}

// -----

// COM: The equations writing z and q form a cycle that is discovered first.
// COM: Starting from the equation writing x, only x[1] reaches it, through the
// COM: equation writing y[1] by means of constant accesses. The cycle passing
// COM: through x[1] and y[1] must be detected without involving the other
// COM: indices of x.

// CHECK-LABEL: @LinkedCyclesThroughConstantAccess

bmodelica.model @LinkedCyclesThroughConstantAccess {
    bmodelica.variable @x : !bmodelica.variable<4x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<4x!bmodelica.real>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @q : !bmodelica.variable<!bmodelica.real>

    // COM: z = 1 - q
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable.get @z : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 1.0>
        %2 = bmodelica.variable.get @q : !bmodelica.real
        %3 = bmodelica.sub %1, %2 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}

    // COM: q = z + x[1]
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %c1 = bmodelica.constant 1 : index
        %0 = bmodelica.variable.get @q : !bmodelica.real
        %1 = bmodelica.variable.get @z : !bmodelica.real
        %2 = bmodelica.variable.get @x : tensor<4x!bmodelica.real>
        %3 = bmodelica.tensor.extract %2[%c1] : tensor<4x!bmodelica.real>
        %4 = bmodelica.add %1, %3 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %5 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}

    // COM: x[i] = 2 - y[i]
    %t2 = bmodelica.equation_template inductions = [%i0] attributes {id = "t2"} {
        %0 = bmodelica.variable.get @x : tensor<4x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 2.0>
        %3 = bmodelica.variable.get @y : tensor<4x!bmodelica.real>
        %4 = bmodelica.tensor.extract %3[%i0] : tensor<4x!bmodelica.real>
        %5 = bmodelica.sub %2, %4 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %6 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
        bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t2"}

    // COM: y[1] = 3 - z (written by means of a constant access)
    %t3 = bmodelica.equation_template inductions = [%i0] attributes {id = "t3"} {
        %c1 = bmodelica.constant 1 : index
        %0 = bmodelica.variable.get @y : tensor<4x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%c1] : tensor<4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 3.0>
        %3 = bmodelica.variable.get @z : !bmodelica.real
        %4 = bmodelica.sub %2, %3 : !bmodelica.real, !bmodelica.real -> !bmodelica.real
        %5 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
        bmodelica.equation_sides %5, %6 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t3"}

    // COM: y[i] = 4
    %t4 = bmodelica.equation_template inductions = [%i0] attributes {id = "t4"} {
        %0 = bmodelica.variable.get @y : tensor<4x!bmodelica.real>
        %1 = bmodelica.tensor.extract %0[%i0] : tensor<4x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica<real 4.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t4"}

    bmodelica.dynamic {
        bmodelica.equation_instance %t2, indices = {[0,3]}, match = <@x, {[0,3]}>
        bmodelica.equation_instance %t0, match = @z
        bmodelica.equation_instance %t1, match = @q
        bmodelica.equation_instance %t3, indices = {[0,0]}, match = <@y, {[1,1]}>
        bmodelica.equation_instance %t4, indices = {[0,0]}, match = <@y, {[0,0]}>
        bmodelica.equation_instance %t4, indices = {[2,3]}, match = <@y, {[2,3]}>
    }

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t0]], match = @z
    // CHECK-DAG: bmodelica.equation_instance %[[t1]], match = @q
    // CHECK-DAG: bmodelica.equation_instance %[[t2]], indices = {[1,1]}, match = <@x, {[1,1]}>
    // CHECK-DAG: bmodelica.equation_instance %[[t3]], indices = {[0,0]}, match = <@y, {[1,1]}>

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t2]], indices = {[0,0],[2,3]}, match = <@x, {[0,0],[2,3]}>

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t4]], indices = {[0,0]}, match = <@y, {[0,0]}>

    // CHECK:     bmodelica.scc
    // CHECK-DAG: bmodelica.equation_instance %[[t4]], indices = {[2,3]}, match = <@y, {[2,3]}>

    // CHECK-NOT: bmodelica.equation_instance

    // COUNT-LABEL: @LinkedCyclesThroughConstantAccess
    // COUNT-COUNT-4: bmodelica.scc
    // COUNT-NOT: bmodelica.scc
}
