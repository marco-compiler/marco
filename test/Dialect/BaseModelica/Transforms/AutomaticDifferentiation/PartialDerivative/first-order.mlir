// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @pder_simpleVar
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.variable_get @pder_x_2
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:      bmodelica.variable_set @pder_y_3, %[[seed_x]]
// CHECK-NEXT:  }

// CHECK-LABEL: @simpleVar_x
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_simpleVar(%[[x]], %[[seed_x]])
// CHECK-NEXT:      bmodelica.variable_set @y, %[[call]]
// CHECK-NEXT:  }

bmodelica.function @simpleVar {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

bmodelica.der_function @simpleVar_x {derivedFunction = @simpleVar, independentVars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulByScalar
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[cst_23:.*]] = bmodelica.constant #bmodelica<real 2.300000e+01>
// CHECK-NEXT:      %[[cst_0:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.variable_get @pder_x_2
// CHECK-NEXT:      %[[mul_cst_23_x:.*]] = bmodelica.mul %[[cst_23]], %[[x]]
// CHECK-NEXT:      %[[mul_cst_0_x:.*]] = bmodelica.mul %[[cst_0]], %[[x]]
// CHECK-NEXT:      %[[mul_cst_23_seed_x:.*]] = bmodelica.mul %[[cst_23]], %[[seed_x]]
// CHECK-NEXT:      %[[add:.*]] = bmodelica.add %[[mul_cst_0_x]], %[[mul_cst_23_seed_x]]
// CHECK-NEXT:      bmodelica.variable_set @y, %[[mul_cst_23_x]]
// CHECK-NEXT:      bmodelica.variable_set @pder_y_3, %[[add]]
// CHECK-NEXT:  }

// CHECK-LABEL: @mulByScalar_x
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_mulByScalar(%[[x]], %[[seed_x]])
// CHECK-NEXT:      bmodelica.variable_set @y, %[[call]]
// CHECK-NEXT:  }

bmodelica.function @mulByScalar {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica<real 23.0>
        %1 = bmodelica.variable_get @x : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @y, %2: !bmodelica.real
    }
}

bmodelica.der_function @mulByScalar_x {derivedFunction = @mulByScalar, independentVars = ["x"]}

// -----

// CHECK-LABEL: @pder_sumOfVars
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_3 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_4 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_z_5 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.variable_get @pder_x_3
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[seed_y:.*]] = bmodelica.variable_get @pder_y_4
// CHECK-NEXT:      %[[add_x_y:.*]] = bmodelica.add %[[x]], %[[y]]
// CHECK-NEXT:      %[[add_seed_x_seed_y:.*]] = bmodelica.add %[[seed_x]], %[[seed_y]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[add_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @pder_z_5, %[[add_seed_x_seed_y]]
// CHECK-NEXT:  }

// CHECK-LABEL: @sumOfVars_x
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[seed_y:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_sumOfVars(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK-NEXT:      bmodelica.variable_set @z, %[[call]]
// CHECK-NEXT:  }

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

bmodelica.der_function @sumOfVars_x {derivedFunction = @sumOfVars, independentVars = ["x"]}

// -----

// CHECK-LABEL: @pder_mulOfVars
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_3 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_4 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_z_5 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.variable_get @pder_x_3
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[seed_y:.*]] = bmodelica.variable_get @pder_y_4
// CHECK-NEXT:      %[[mul_x_y:.*]] = bmodelica.mul %[[x]], %[[y]]
// CHECK-NEXT:      %[[mul_seed_x_y:.*]] = bmodelica.mul %[[seed_x]], %[[y]]
// CHECK-NEXT:      %[[mul_x_seed_y:.*]] = bmodelica.mul %[[x]], %[[seed_y]]
// CHECK-NEXT:      %[[add:.*]] = bmodelica.add %[[mul_seed_x_y]], %[[mul_x_seed_y]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[mul_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @pder_z_5, %[[add]]
// CHECK-NEXT:  }

// CHECK-LABEL: @mulOfVars_x
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[seed_x:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[seed_y:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_mulOfVars(%[[x]], %[[y]], %[[seed_x]], %[[seed_y]])
// CHECK-NEXT:      bmodelica.variable_set @z, %[[call]]
// CHECK-NEXT:  }

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

bmodelica.der_function @mulOfVars_x {derivedFunction = @mulOfVars, independentVars = ["x"]}

// -----

// CHECK-LABEL: @pder_scalarMul
// CHECK-NEXT:  bmodelica.variable @x1 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y1 : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x1_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y1_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x1:.*]] = bmodelica.variable_get @x1
// CHECK-NEXT:      %[[seed_x1:.*]] = bmodelica.variable_get @pder_x1_2
// CHECK-NEXT:      %[[cst_23:.*]] = bmodelica.constant #bmodelica<real 2.300000e+01>
// CHECK-NEXT:      %[[cst_0:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[mul_x1_cst_23:.*]] = bmodelica.mul %[[x1]], %[[cst_23]]
// CHECK-NEXT:      %[[mul_seed_x1_cst_23:.*]] = bmodelica.mul %[[seed_x1]], %[[cst_23]]
// CHECK-NEXT:      %[[mul_x1_cst_0:.*]] = bmodelica.mul %[[x1]], %[[cst_0]]
// CHECK-NEXT:      %[[add:.*]] = bmodelica.add %[[mul_seed_x1_cst_23]], %[[mul_x1_cst_0]]
// CHECK-NEXT:      bmodelica.variable_set @y1, %[[mul_x1_cst_23]]
// CHECK-NEXT:      bmodelica.variable_set @pder_y1_3, %[[add]]
// CHECK-NEXT:  }

// CHECK-LABEL: @pder_callOpDer
// CHECK-NEXT:  bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x2_2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y2_3 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[cst_57:.*]] = bmodelica.constant #bmodelica<int 57>
// CHECK-NEXT:      %[[cst_0:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[x2:.*]] = bmodelica.variable_get @x2
// CHECK-NEXT:      %[[seed_x2:.*]] = bmodelica.variable_get @pder_x2_2
// CHECK-NEXT:      %[[mul_cst_57_x2:.*]] = bmodelica.mul %[[cst_57]], %[[x2]]
// CHECK-NEXT:      %[[mul_cst_0_x2:.*]] = bmodelica.mul %[[cst_0]], %[[x2]]
// CHECK-NEXT:      %[[mul_cst_57_seed_x2:.*]] = bmodelica.mul %[[cst_57]], %[[seed_x2]]
// CHECK-NEXT:      %[[add:.*]] = bmodelica.add %[[mul_cst_0_x2]], %[[mul_cst_57_seed_x2]]
// CHECK-NEXT:      %[[call_scalarMul:.*]] = bmodelica.call @scalarMul(%[[mul_cst_57_x2]])
// CHECK-NEXT:      %[[call_pder_scalarMul:.*]] = bmodelica.call @pder_scalarMul(%[[mul_cst_57_x2]], %[[add]])
// CHECK-NEXT:      bmodelica.variable_set @y2, %[[call_scalarMul]]
// CHECK-NEXT:      bmodelica.variable_set @pder_y2_3, %[[call_pder_scalarMul]]
// CHECK-NEXT:  }

// CHECK-LABEL: @callOpDer_x2
// CHECK-NEXT:  bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x2:.*]] = bmodelica.variable_get @x2
// CHECK-NEXT:      %[[seed_x2:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_callOpDer(%[[x2]], %[[seed_x2]])
// CHECK-NEXT:      bmodelica.variable_set @y2, %[[call]]
// CHECK-NEXT:  }

bmodelica.function @scalarMul {
    bmodelica.variable @x1 : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y1 : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x1 : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<real 23.0>
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @y1, %2: !bmodelica.real
    }
}

bmodelica.function @callOpDer {
    bmodelica.variable @x2 : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y2 : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.constant #bmodelica<int 57>
        %1 = bmodelica.variable_get @x2 : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.call @scalarMul(%2) : (!bmodelica.real) -> (!bmodelica.real)
        bmodelica.variable_set @y2, %3: !bmodelica.real
    }
}

bmodelica.der_function @callOpDer_x2 {derivedFunction = @callOpDer, independentVars = ["x2"]}
