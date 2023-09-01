// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @pder_simpleVar_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x_2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y_3 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.variable_get @pder_1_x_2
// CHECK:       modelica.variable_set @y, %[[x]]
// CHECK:       modelica.variable_set @pder_1_y_3, %[[seed_x]]
// CHECK:   }

// CHECK-LABEL: @simpleVar_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = modelica.call @pder_simpleVar_x(%[[x]], %[[seed_x]])
// CHECK:       modelica.variable_set @y, %[[call]]
// CHECK:   }

modelica.function @simpleVar {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        modelica.variable_set @y, %0 : !modelica.real
    }
}

modelica.der_function @simpleVar_x {derived_function = "simpleVar", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulByScalar_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x_2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y_3 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[cst_23:.*]] = modelica.constant #modelica.real<2.300000e+01>
// CHECK:       %[[cst_0:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.variable_get @pder_1_x_2
// CHECK:       %[[mul_cst_23_x:.*]] = modelica.mul %[[cst_23]], %[[x]]
// CHECK:       %[[mul_cst_0_x:.*]] = modelica.mul %[[cst_0]], %[[x]]
// CHECK:       %[[mul_cst_23_seed_x:.*]] = modelica.mul %[[cst_23]], %[[seed_x]]
// CHECK:       %[[add:.*]] = modelica.add %[[mul_cst_0_x]], %[[mul_cst_23_seed_x]]
// CHECK:       modelica.variable_set @y, %[[mul_cst_23_x]]
// CHECK:       modelica.variable_set @pder_1_y_3, %[[add]]
// CHECK:   }

// CHECK-LABEL: @mulByScalar_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = modelica.call @pder_mulByScalar_x(%[[x]], %[[seed_x]])
// CHECK:       modelica.variable_set @y, %[[call]]
// CHECK:   }

modelica.function @mulByScalar {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.constant #modelica.real<23.0>
        %1 = modelica.variable_get @x : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @y, %2: !modelica.real
    }
}

modelica.der_function @mulByScalar_x {derived_function = "mulByScalar", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_sumOfVars_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x_3 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y_4 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_z_5 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.variable_get @pder_1_x_3
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[seed_y:.*]] = modelica.variable_get @pder_1_y_4
// CHECK:       %[[add_x_y:.*]] = modelica.add %[[x]], %[[y]]
// CHECK:       %[[add_seed_x_seed_y:.*]] = modelica.add %[[seed_x]], %[[seed_y]]
// CHECK:       modelica.variable_set @z, %[[add_x_y]]
// CHECK:       modelica.variable_set @pder_1_z_5, %[[add_seed_x_seed_y]]
// CHECK:   }

// CHECK-LABEL: @sumOfVars_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[seed_x:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:       %[[seed_y:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK:       %[[call:.*]] = modelica.call @pder_sumOfVars_x(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK:       modelica.variable_set @z, %[[call]]
// CHECK:   }

modelica.function @sumOfVars {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2: !modelica.real
    }
}

modelica.der_function @sumOfVars_x {derived_function = "sumOfVars", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulOfVars_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x_3 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y_4 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_z_5 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[seed_x:.*]] = modelica.variable_get @pder_1_x_3
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[seed_y:.*]] = modelica.variable_get @pder_1_y_4
// CHECK:       %[[mul_x_y:.*]] = modelica.mul %[[x]], %[[y]]
// CHECK:       %[[mul_seed_x_y:.*]] = modelica.mul %[[seed_x]], %[[y]]
// CHECK:       %[[mul_x_seed_y:.*]] = modelica.mul %[[x]], %[[seed_y]]
// CHECK:       %[[add:.*]] = modelica.add %[[mul_seed_x_y]], %[[mul_x_seed_y]]
// CHECK:       modelica.variable_set @z, %[[mul_x_y]]
// CHECK:       modelica.variable_set @pder_1_z_5, %[[add]]
// CHECK:   }

// CHECK-LABEL: @mulOfVars_x
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[seed_x:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:       %[[seed_y:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK:       %[[call:.*]] = modelica.call @pder_mulOfVars_x(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK:       modelica.variable_set @z, %[[call]]
// CHECK:   }

modelica.function @mulOfVars {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2: !modelica.real
    }
}

modelica.der_function @mulOfVars_x {derived_function = "mulOfVars", independent_vars = ["x"]}

// -----

// CHECK-LABEL: @call_pder_scalarMul
// CHECK:   modelica.variable @x1 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y1 : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x1_2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y1_3 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x1:.*]] = modelica.variable_get @x1
// CHECK:       %[[seed_x1:.*]] = modelica.variable_get @pder_1_x1_2
// CHECK:       %[[cst_23:.*]] = modelica.constant #modelica.real<2.300000e+01>
// CHECK:       %[[cst_0:.*]] = modelica.constant #modelica.real<0.000000e+00>
// CHECK:       %[[mul_x1_cst_23:.*]] = modelica.mul %[[x1]], %[[cst_23]]
// CHECK:       %[[mul_seed_x1_cst_23:.*]] = modelica.mul %[[seed_x1]], %[[cst_23]]
// CHECK:       %[[mul_x1_cst_0:.*]] = modelica.mul %[[x1]], %[[cst_0]]
// CHECK:       %[[add:.*]] = modelica.add %[[mul_seed_x1_cst_23]], %[[mul_x1_cst_0]]
// CHECK:       modelica.variable_set @y1, %[[mul_x1_cst_23]]
// CHECK:       modelica.variable_set @pder_1_y1_3, %[[add]]
// CHECK:   }

// CHECK-LABEL: @pder_callOpDer_x2
// CHECK:   modelica.variable @x2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y2 : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @pder_1_x2_2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @pder_1_y2_3 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[cst_57:.*]] = modelica.constant #modelica.int<57>
// CHECK:       %[[cst_0:.*]] = modelica.constant #modelica.int<0>
// CHECK:       %[[x2:.*]] = modelica.variable_get @x2
// CHECK:       %[[seed_x2:.*]] = modelica.variable_get @pder_1_x2_2
// CHECK:       %[[mul_cst_57_x2:.*]] = modelica.mul %[[cst_57]], %[[x2]]
// CHECK:       %[[mul_cst_0_x2:.*]] = modelica.mul %[[cst_0]], %[[x2]]
// CHECK:       %[[mul_cst_57_seed_x2:.*]] = modelica.mul %[[cst_57]], %[[seed_x2]]
// CHECK:       %[[add:.*]] = modelica.add %[[mul_cst_0_x2]], %[[mul_cst_57_seed_x2]]
// CHECK:       %[[call_scalarMul:.*]] = modelica.call @scalarMul(%[[mul_cst_57_x2]])
// CHECK:       %[[call_pder_scalarMul:.*]] = modelica.call @call_pder_scalarMul(%[[mul_cst_57_x2]], %[[add]])
// CHECK:       modelica.variable_set @y2, %[[call_scalarMul]]
// CHECK:       modelica.variable_set @pder_1_y2_3, %[[call_pder_scalarMul]]
// CHECK:   }

// CHECK-LABEL: @callOpDer_x2
// CHECK:   modelica.variable @x2 : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y2 : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x2:.*]] = modelica.variable_get @x2
// CHECK:       %[[seed_x2:.*]] = modelica.constant #modelica.real<1.000000e+00>
// CHECK:       %[[call:.*]] = modelica.call @pder_callOpDer_x2(%[[x2]], %[[seed_x2]])
// CHECK:       modelica.variable_set @y2, %[[call]]
// CHECK:   }

modelica.function @scalarMul {
    modelica.variable @x1 : !modelica.variable<!modelica.real, input>
    modelica.variable @y1 : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x1 : !modelica.real
        %1 = modelica.constant #modelica.real<23.0>
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @y1, %2: !modelica.real
    }
}

modelica.function @callOpDer {
    modelica.variable @x2 : !modelica.variable<!modelica.real, input>
    modelica.variable @y2 : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.constant #modelica.int<57>
        %1 = modelica.variable_get @x2 : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        %3 = modelica.call @scalarMul(%2) : (!modelica.real) -> (!modelica.real)
        modelica.variable_set @y2, %3: !modelica.real
    }
}

modelica.der_function @callOpDer_x2 {derived_function = "callOpDer", independent_vars = ["x2"]}
