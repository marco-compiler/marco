// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @var_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       modelica.variable_set @y, %[[x]]
// CHECK:       modelica.variable_set @der_y, %[[der_x]]
// CHECK:   }

modelica.function @var attributes {derivative = #modelica.derivative<"var_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        modelica.variable_set @y, %0 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @neg_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[neg_x:.*]] = modelica.neg %[[x]]
// CHECK:       %[[neg_der_x:.*]] = modelica.neg %[[der_x]]
// CHECK:       modelica.variable_set @y, %[[neg_x]]
// CHECK:       modelica.variable_set @der_y, %[[neg_der_x]]
// CHECK:   }

modelica.function @neg attributes {derivative = #modelica.derivative<"neg_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.neg %0 : !modelica.real -> !modelica.real
        modelica.variable_set @y, %1 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @add_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[der_y:.*]] = modelica.variable_get @der_y
// CHECK:       %[[add_x_y:.*]] = modelica.add %[[x]], %[[y]]
// CHECK:       %[[add_der_x_der_y:.*]] = modelica.add %[[der_x]], %[[der_y]]
// CHECK:       modelica.variable_set @z, %[[add_x_y]]
// CHECK:       modelica.variable_set @der_z, %[[add_der_x_der_y]]
// CHECK:   }

modelica.function @add attributes {derivative = #modelica.derivative<"add_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @sub_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_z : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[der_y:.*]] = modelica.variable_get @der_y
// CHECK:       %[[sub_x_y:.*]] = modelica.sub %[[x]], %[[y]]
// CHECK:       %[[sub_der_x_der_y:.*]] = modelica.sub %[[der_x]], %[[der_y]]
// CHECK:       modelica.variable_set @z, %[[sub_x_y]]
// CHECK:       modelica.variable_set @der_z, %[[sub_der_x_der_y]]
// CHECK:   }

modelica.function @sub attributes {derivative = #modelica.derivative<"sub_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.sub %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @mul_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_z : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[der_y:.*]] = modelica.variable_get @der_y
// CHECK:       %[[mul_x_y:.*]] = modelica.mul %[[x]], %[[y]]
// CHECK:       %[[mul_der_x_y:.*]] = modelica.mul %[[der_x]], %[[y]]
// CHECK:       %[[mul_x_der_y:.*]] = modelica.mul %[[x]], %[[der_y]]
// CHECK:       %[[add:.*]] = modelica.add %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK:       modelica.variable_set @z, %[[mul_x_y]]
// CHECK:       modelica.variable_set @der_z, %[[add]]
// CHECK:   }

modelica.function @mul attributes {derivative = #modelica.derivative<"mul_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.mul %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}

// -----

// CHECK-LABEL: @div_der
// CHECK:   modelica.variable @x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @z : !modelica.variable<!modelica.real>
// CHECK:   modelica.variable @der_x : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_y : !modelica.variable<!modelica.real, input>
// CHECK:   modelica.variable @der_z : !modelica.variable<!modelica.real, output>
// CHECK:   modelica.algorithm {
// CHECK:       %[[x:.*]] = modelica.variable_get @x
// CHECK:       %[[der_x:.*]] = modelica.variable_get @der_x
// CHECK:       %[[y:.*]] = modelica.variable_get @y
// CHECK:       %[[der_y:.*]] = modelica.variable_get @der_y
// CHECK:       %[[div_x_y:.*]] = modelica.div %[[x]], %[[y]]
// CHECK:       %[[mul_der_x_y:.*]] = modelica.mul %[[der_x]], %[[y]]
// CHECK:       %[[mul_x_der_y:.*]] = modelica.mul %[[x]], %[[der_y]]
// CHECK:       %[[sub:.*]] = modelica.sub %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK:       %[[exponent:.*]] = modelica.constant #modelica.real<2.000000e+00>
// CHECK:       %[[pow:.*]] = modelica.pow %2, %[[exponent]]
// CHECK:       %[[der_div:.*]] = modelica.div %[[sub]], %[[pow]]
// CHECK:       modelica.variable_set @z, %[[div_x_y]]
// CHECK:       modelica.variable_set @der_z, %[[der_div]]
// CHECK:   }

modelica.function @div attributes {derivative = #modelica.derivative<"div_der", 1>} {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, input>
    modelica.variable @z : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        %1 = modelica.variable_get @y : !modelica.real
        %2 = modelica.div %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        modelica.variable_set @z, %2 : !modelica.real
    }
}
