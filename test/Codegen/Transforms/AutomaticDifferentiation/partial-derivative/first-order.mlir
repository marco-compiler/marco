// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @pder_simpleVar_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.variable_get @pder_1_x_2
// CHECK:       bmodelica.variable_set @y, %[[x]]
// CHECK:       bmodelica.variable_set @pder_1_y_3, %[[seed_x]]
// CHECK:   }

// CHECK-LABEL: @simpleVar_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = bmodelica.call @pder_simpleVar_x(%[[x]], %[[seed_x]])
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

bmodelica.function @simpleVar {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

bmodelica.der_function @simpleVar_x {derived_function = "simpleVar", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulByScalar_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[cst_23:.*]] = bmodelica.constant #bmodelica.real<2.300000e+01>
// CHECK:       %[[cst_0:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.variable_get @pder_1_x_2
// CHECK:       %[[mul_cst_23_x:.*]] = bmodelica.mul %[[cst_23]], %[[x]]
// CHECK:       %[[mul_cst_0_x:.*]] = bmodelica.mul %[[cst_0]], %[[x]]
// CHECK:       %[[mul_cst_23_seed_x:.*]] = bmodelica.mul %[[cst_23]], %[[seed_x]]
// CHECK:       %[[add:.*]] = bmodelica.add %[[mul_cst_0_x]], %[[mul_cst_23_seed_x]]
// CHECK:       bmodelica.variable_set @y, %[[mul_cst_23_x]]
// CHECK:       bmodelica.variable_set @pder_1_y_3, %[[add]]
// CHECK:   }

// CHECK-LABEL: @mulByScalar_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = bmodelica.call @pder_mulByScalar_x(%[[x]], %[[seed_x]])
// CHECK:       bmodelica.variable_set @y, %[[call]]
// CHECK:   }

bmodelica.function @mulByScalar {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica.real<23.0>
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @y, %2: !bmodelica.real
    }
}

bmodelica.der_function @mulByScalar_x {derived_function = "mulByScalar", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_sumOfVars_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x_3 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y_4 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_z_5 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.variable_get @pder_1_x_3
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[seed_y:.*]] = bmodelica.variable_get @pder_1_y_4
// CHECK:       %[[add_x_y:.*]] = bmodelica.add %[[x]], %[[y]]
// CHECK:       %[[add_seed_x_seed_y:.*]] = bmodelica.add %[[seed_x]], %[[seed_y]]
// CHECK:       bmodelica.variable_set @z, %[[add_x_y]]
// CHECK:       bmodelica.variable_set @pder_1_z_5, %[[add_seed_x_seed_y]]
// CHECK:   }

// CHECK-LABEL: @sumOfVars_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[seed_x:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:       %[[seed_y:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK:       %[[call:.*]] = bmodelica.call @pder_sumOfVars_x(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK:       bmodelica.variable_set @z, %[[call]]
// CHECK:   }

bmodelica.function @sumOfVars {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2: !bmodelica.real
    }
}

bmodelica.der_function @sumOfVars_x {derived_function = "sumOfVars", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulOfVars_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x_3 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y_4 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_z_5 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = bmodelica.variable_get @pder_1_x_3
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[seed_y:.*]] = bmodelica.variable_get @pder_1_y_4
// CHECK:       %[[mul_x_y:.*]] = bmodelica.mul %[[x]], %[[y]]
// CHECK:       %[[mul_seed_x_y:.*]] = bmodelica.mul %[[seed_x]], %[[y]]
// CHECK:       %[[mul_x_seed_y:.*]] = bmodelica.mul %[[x]], %[[seed_y]]
// CHECK:       %[[add:.*]] = bmodelica.add %[[mul_seed_x_y]], %[[mul_x_seed_y]]
// CHECK:       bmodelica.variable_set @z, %[[mul_x_y]]
// CHECK:       bmodelica.variable_set @pder_1_z_5, %[[add]]
// CHECK:   }

// CHECK-LABEL: @mulOfVars_x
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[seed_x:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:       %[[seed_y:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK:       %[[call:.*]] = bmodelica.call @pder_mulOfVars_x(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK:       bmodelica.variable_set @z, %[[call]]
// CHECK:   }

bmodelica.function @mulOfVars {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2: !bmodelica.real
    }
}

bmodelica.der_function @mulOfVars_x {derived_function = "mulOfVars", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @call_pder_scalarMul
// CHECK:   bmodelica.variable @x1 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y1 : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x1_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y1_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x1:.*]] = bmodelica.variable_get @x1
// CHECK:       %[[seed_x1:.*]] = bmodelica.variable_get @pder_1_x1_2
// CHECK:       %[[cst_23:.*]] = bmodelica.constant #bmodelica.real<2.300000e+01>
// CHECK:       %[[cst_0:.*]] = bmodelica.constant #bmodelica.real<0.000000e+00>
// CHECK:       %[[mul_x1_cst_23:.*]] = bmodelica.mul %[[x1]], %[[cst_23]]
// CHECK:       %[[mul_seed_x1_cst_23:.*]] = bmodelica.mul %[[seed_x1]], %[[cst_23]]
// CHECK:       %[[mul_x1_cst_0:.*]] = bmodelica.mul %[[x1]], %[[cst_0]]
// CHECK:       %[[add:.*]] = bmodelica.add %[[mul_seed_x1_cst_23]], %[[mul_x1_cst_0]]
// CHECK:       bmodelica.variable_set @y1, %[[mul_x1_cst_23]]
// CHECK:       bmodelica.variable_set @pder_1_y1_3, %[[add]]
// CHECK:   }

// CHECK-LABEL: @pder_callOpDer_x2
// CHECK:   bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @pder_1_x2_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @pder_1_y2_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[cst_57:.*]] = bmodelica.constant #bmodelica.int<57>
// CHECK:       %[[cst_0:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK:       %[[x2:.*]] = bmodelica.variable_get @x2
// CHECK:       %[[seed_x2:.*]] = bmodelica.variable_get @pder_1_x2_2
// CHECK:       %[[mul_cst_57_x2:.*]] = bmodelica.mul %[[cst_57]], %[[x2]]
// CHECK:       %[[mul_cst_0_x2:.*]] = bmodelica.mul %[[cst_0]], %[[x2]]
// CHECK:       %[[mul_cst_57_seed_x2:.*]] = bmodelica.mul %[[cst_57]], %[[seed_x2]]
// CHECK:       %[[add:.*]] = bmodelica.add %[[mul_cst_0_x2]], %[[mul_cst_57_seed_x2]]
// CHECK:       %[[call_scalarMul:.*]] = bmodelica.call @scalarMul(%[[mul_cst_57_x2]])
// CHECK:       %[[call_pder_scalarMul:.*]] = bmodelica.call @call_pder_scalarMul(%[[mul_cst_57_x2]], %[[add]])
// CHECK:       bmodelica.variable_set @y2, %[[call_scalarMul]]
// CHECK:       bmodelica.variable_set @pder_1_y2_3, %[[call_pder_scalarMul]]
// CHECK:   }

// CHECK-LABEL: @callOpDer_x2
// CHECK:   bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x2:.*]] = bmodelica.variable_get @x2
// CHECK:       %[[seed_x2:.*]] = bmodelica.constant #bmodelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = bmodelica.call @pder_callOpDer_x2(%[[x2]], %[[seed_x2]])
// CHECK:       bmodelica.variable_set @y2, %[[call]]
// CHECK:   }

bmodelica.function @scalarMul {
    bmodelica.variable @x1 : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y1 : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x1 : !bmodelica.real
        %1 = bmodelica.constant #bmodelica.real<23.0>
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @y1, %2: !bmodelica.real
    }
}

bmodelica.function @callOpDer {
    bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica.int<57>
        %1 = bmodelica.variable_get @x2 : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.call @scalarMul(%2) : (!bmodelica.real) -> (!bmodelica.real)
        bmodelica.variable_set @y2, %3: !bmodelica.real
    }
}

bmodelica.der_function @callOpDer_x2 {derived_function = "callOpDer", independent_vars = ["x2"]}
