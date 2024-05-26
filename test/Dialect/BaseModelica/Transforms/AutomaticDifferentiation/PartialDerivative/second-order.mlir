// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @pder_pder_mulOfVars
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_3 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_4 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_z_5 : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_x_6 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_y_7 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_z_8 : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @pder_pder_x_3_9 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_pder_y_4_10 : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @pder_pder_z_5_11 : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x : !bmodelica.real
// CHECK-NEXT:      %[[x_dy:.*]] = bmodelica.variable_get @pder_x_6 : !bmodelica.real
// CHECK-NEXT:      %[[x_dx:.*]] = bmodelica.variable_get @pder_x_3 : !bmodelica.real
// CHECK-NEXT:      %[[x_dx_dy:.*]] = bmodelica.variable_get @pder_pder_x_3_9 : !bmodelica.real
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y : !bmodelica.real
// CHECK-NEXT:      %[[y_dx_dy:.*]] = bmodelica.variable_get @pder_y_7 : !bmodelica.real
// CHECK-NEXT:      %[[y_dx:.*]] = bmodelica.variable_get @pder_y_4 : !bmodelica.real
// CHECK-NEXT:      %[[x_dy_dy:.*]] = bmodelica.variable_get @pder_pder_y_4_10 : !bmodelica.real
// CHECK-NEXT:      %[[z_result:.*]] = bmodelica.mul %[[x]], %[[y]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_dy_y:.*]] = bmodelica.mul %[[x_dy]], %[[y]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_y_dx_dy:.*]] = bmodelica.mul %[[x]], %[[y_dx_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[z_dy_result:.*]] = bmodelica.add %[[mul_x_dy_y]], %[[mul_x_y_dx_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_dx_y:.*]] = bmodelica.mul %[[x_dx]], %[[y]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_dx_dy_y:.*]] = bmodelica.mul %[[x_dx_dy]], %[[y]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_dx_y_dx_dy:.*]] = bmodelica.mul %[[x_dx]], %[[y_dx_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[add_mul_x_dx_dy_y_mul_x_dx_y_dx_dy:.*]] = bmodelica.add %[[mul_x_dx_dy_y]], %[[mul_x_dx_y_dx_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_y_dx:.*]] = bmodelica.mul %[[x]], %[[y_dx]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_dy_y_dx:.*]] = bmodelica.mul %[[x_dy]], %[[y_dx]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[mul_x_x_dy_dy:.*]] = bmodelica.mul %[[x]], %[[x_dy_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[add_mul_x_dy_y_dx_mul_x_x_dy_dy:.*]] = bmodelica.add %[[mul_x_dy_y_dx]], %[[mul_x_x_dy_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[z_dx_result:.*]] = bmodelica.add %[[mul_x_dx_y]], %[[mul_x_y_dx]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      %[[z_dx_dy_result:.*]] = bmodelica.add %[[add_mul_x_dx_dy_y_mul_x_dx_y_dx_dy]], %[[add_mul_x_dy_y_dx_mul_x_x_dy_dy]] : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
// CHECK-NEXT:      bmodelica.variable_set @z, %[[z_result]] : !bmodelica.real
// CHECK-NEXT:      bmodelica.variable_set @pder_z_8, %[[z_dy_result]] : !bmodelica.real
// CHECK-NEXT:      bmodelica.variable_set @pder_z_5, %[[z_dx_result]] : !bmodelica.real
// CHECK-NEXT:      bmodelica.variable_set @pder_pder_z_5_11, %[[z_dx_dy_result]] : !bmodelica.real
// CHECK-NEXT:  }

// CHECK-LABEL: @mulOfVars_x_y
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[x_dx:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[y_dx:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[x_dy:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[y_dy:.*]] = bmodelica.constant #bmodelica<real 1.000000e+00>
// CHECK-NEXT:      %[[x_dx_dy:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[y_dx_dy:.*]] = bmodelica.constant #bmodelica<real 0.000000e+00>
// CHECK-NEXT:      %[[call:.*]] = bmodelica.call @pder_pder_mulOfVars(%[[x]], %[[y]], %[[x_dx]], %[[y_dx]], %[[x_dy]], %[[y_dy]], %[[x_dx_dy]], %[[y_dx_dy]])
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

bmodelica.der_function @mulOfVars_x_y {derivedFunction = @mulOfVars, independentVars = ["x", "y"]}
