// RUN: modelica-opt %s --split-input-file --auto-diff | FileCheck %s

// CHECK-LABEL: @var_der
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      bmodelica.variable_set @y, %[[x]]
// CHECK-NEXT:      bmodelica.variable_set @der_y, %[[der_x]]
// CHECK-NEXT:  }

bmodelica.function @var attributes {derivative = #bmodelica<func_der "var_der", 1>} {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

// -----

// CHECK-LABEL: @neg_der
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[neg_x:.*]] = bmodelica.neg %[[x]]
// CHECK-NEXT:      %[[neg_der_x:.*]] = bmodelica.neg %[[der_x]]
// CHECK-NEXT:      bmodelica.variable_set @y, %[[neg_x]]
// CHECK-NEXT:      bmodelica.variable_set @der_y, %[[neg_der_x]]
// CHECK-NEXT:  }

bmodelica.function @neg attributes {derivative = #bmodelica<func_der "neg_der", 1>} {
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
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK-NEXT:      %[[add_x_y:.*]] = bmodelica.add %[[x]], %[[y]]
// CHECK-NEXT:      %[[add_der_x_der_y:.*]] = bmodelica.add %[[der_x]], %[[der_y]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[add_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_z, %[[add_der_x_der_y]]
// CHECK-NEXT:  }

bmodelica.function @add attributes {derivative = #bmodelica<func_der "add_der", 1>} {
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
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK-NEXT:      %[[sub_x_y:.*]] = bmodelica.sub %[[x]], %[[y]]
// CHECK-NEXT:      %[[sub_der_x_der_y:.*]] = bmodelica.sub %[[der_x]], %[[der_y]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[sub_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_z, %[[sub_der_x_der_y]]
// CHECK-NEXT:  }

bmodelica.function @sub attributes {derivative = #bmodelica<func_der "sub_der", 1>} {
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
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK-NEXT:      %[[mul_x_y:.*]] = bmodelica.mul %[[x]], %[[y]]
// CHECK-NEXT:      %[[mul_der_x_y:.*]] = bmodelica.mul %[[der_x]], %[[y]]
// CHECK-NEXT:      %[[mul_x_der_y:.*]] = bmodelica.mul %[[x]], %[[der_y]]
// CHECK-NEXT:      %[[add:.*]] = bmodelica.add %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[mul_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_z, %[[add]]
// CHECK-NEXT:  }

bmodelica.function @mul attributes {derivative = #bmodelica<func_der "mul_der", 1>} {
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
// CHECK-NEXT:  bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @z : !bmodelica.variable<!bmodelica.real>
// CHECK-NEXT:  bmodelica.variable @der_x : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_y : !bmodelica.variable<!bmodelica.real, input>
// CHECK-NEXT:  bmodelica.variable @der_z : !bmodelica.variable<!bmodelica.real, output>
// CHECK-NEXT:  bmodelica.algorithm {
// CHECK-NEXT:      %[[x:.*]] = bmodelica.variable_get @x
// CHECK-NEXT:      %[[der_x:.*]] = bmodelica.variable_get @der_x
// CHECK-NEXT:      %[[y:.*]] = bmodelica.variable_get @y
// CHECK-NEXT:      %[[der_y:.*]] = bmodelica.variable_get @der_y
// CHECK-NEXT:      %[[div_x_y:.*]] = bmodelica.div %[[x]], %[[y]]
// CHECK-NEXT:      %[[mul_der_x_y:.*]] = bmodelica.mul %[[der_x]], %[[y]]
// CHECK-NEXT:      %[[mul_x_der_y:.*]] = bmodelica.mul %[[x]], %[[der_y]]
// CHECK-NEXT:      %[[sub:.*]] = bmodelica.sub %[[mul_der_x_y]], %[[mul_x_der_y]]
// CHECK-NEXT:      %[[exponent:.*]] = bmodelica.constant #bmodelica<real 2.000000e+00>
// CHECK-NEXT:      %[[pow:.*]] = bmodelica.pow %2, %[[exponent]]
// CHECK-NEXT:      %[[der_div:.*]] = bmodelica.div %[[sub]], %[[pow]]
// CHECK-NEXT:      bmodelica.variable_set @z, %[[div_x_y]]
// CHECK-NEXT:      bmodelica.variable_set @der_z, %[[der_div]]
// CHECK-NEXT:  }

bmodelica.function @div attributes {derivative = #bmodelica<func_der "div_der", 1>} {
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
