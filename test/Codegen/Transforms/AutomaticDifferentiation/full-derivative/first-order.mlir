// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @var_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       bmodelica.variable_set @y, %[[x]]
// CHECK:       bmodelica.variable_set @der_y, %[[der_x]]
// CHECK:   }

bmodelica.function @var attributes {derivative = #bmodelica.derivative<"var_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @neg_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[neg_x:.*]] = bmodelica.neg %[[x]]
// CHECK:       %[[neg_der_x:.*]] = bmodelica.neg %[[der_x]]
// CHECK:       bmodelica.variable_set @y, %[[neg_x]]
// CHECK:       bmodelica.variable_set @der_y, %[[neg_der_x]]
// CHECK:   }

bmodelica.function @neg attributes {derivative = #bmodelica.derivative<"neg_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.neg %0 : !bmodelica.real -> !bmodelica.real
        bmodelica.variable_set @y, %1 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @add_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK:       %[[add_x_y:.*]] = bmodelica.add %[[x]], %[[y]]
// CHECK:       %[[add_der_x_der_y:.*]] = bmodelica.add %[[der_x]], %[[der_y]]
// CHECK:       bmodelica.variable_set @z, %[[add_x_y]]
// CHECK:       bmodelica.variable_set @der_z, %[[add_der_x_der_y]]
// CHECK:   }

bmodelica.function @add attributes {derivative = #bmodelica.derivative<"add_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @sub_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK:       %[[sub_x_y:.*]] = bmodelica.sub %[[x]], %[[y]]
// CHECK:       %[[sub_der_x_der_y:.*]] = bmodelica.sub %[[der_x]], %[[der_y]]
// CHECK:       bmodelica.variable_set @z, %[[sub_x_y]]
// CHECK:       bmodelica.variable_set @der_z, %[[sub_der_x_der_y]]
// CHECK:   }

bmodelica.function @sub attributes {derivative = #bmodelica.derivative<"sub_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.sub %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @mul_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK:       %[[mul_x_y:.*]] = bmodelica.mul %[[x]], %[[y]]
// CHECK:       %[[mul_der_x_y:.*]] = bmodelica.mul %[[der_x]], %[[y]]
// CHECK:       %[[mul_x_der_y:.*]] = bmodelica.mul %[[x]], %[[der_y]]
// CHECK:       %[[add:.*]] = bmodelica.add %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK:       bmodelica.variable_set @z, %[[mul_x_y]]
// CHECK:       bmodelica.variable_set @der_z, %[[add]]
// CHECK:   }

bmodelica.function @mul attributes {derivative = #bmodelica.derivative<"mul_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.mul %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @div_der
// CHECK:   bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK:   bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK:   bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK:   bmodelica.algorithm {
// CHECK:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK:       %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK:       %[[y:.*]] = bmodelica.variable_get @y
// CHECK:       %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK:       %[[div_x_y:.*]] = bmodelica.div %[[x]], %[[y]]
// CHECK:       %[[mul_der_x_y:.*]] = bmodelica.mul %[[der_x]], %[[y]]
// CHECK:       %[[mul_x_der_y:.*]] = bmodelica.mul %[[x]], %[[der_y]]
// CHECK:       %[[sub:.*]] = bmodelica.sub %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK:       %[[exponent:.*]] = bmodelica.constant #bmodelica.real<2.000000e+00>
// CHECK:       %[[pow:.*]] = bmodelica.pow %2, %[[exponent]]
// CHECK:       %[[der_div:.*]] = bmodelica.div %[[sub]], %[[pow]]
// CHECK:       bmodelica.variable_set @z, %[[div_x_y]]
// CHECK:       bmodelica.variable_set @der_z, %[[der_div]]
// CHECK:   }

bmodelica.function @div attributes {derivative = #bmodelica.derivative<"div_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @z : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.variable_get @y : !bmodelica.real
        %2 = bmodelica.div %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        bmodelica.variable_set @z, %2 : !bmodelica.real
    }
}
