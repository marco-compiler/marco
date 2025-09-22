// RUN: modelica-opt %s --split-input-file --pantelides | FileCheck %s

module {
  bmodelica.model @Pendulum der = [<@x, @der_x>, <@y, @der_y>, <@u, @der_u>, <@v, @der_v>] {
    bmodelica.variable @x : !bmodelica.variable<f64>
    bmodelica.variable @y : !bmodelica.variable<f64>
    bmodelica.variable @u : !bmodelica.variable<f64>
    bmodelica.variable @v : !bmodelica.variable<f64>
    bmodelica.variable @lambda : !bmodelica.variable<f64>
    bmodelica.variable @g : !bmodelica.variable<f64, parameter>
    bmodelica.variable @L : !bmodelica.variable<f64, parameter>
    bmodelica.variable @der_x : !bmodelica.variable<f64>
    bmodelica.variable @der_y : !bmodelica.variable<f64>
    bmodelica.variable @der_u : !bmodelica.variable<f64>
    bmodelica.variable @der_v : !bmodelica.variable<f64>

    // CHECK-DAG: bmodelica.variable @x
    // CHECK-DAG: bmodelica.variable @y
    // CHECK-DAG: bmodelica.variable @u
    // CHECK-DAG: bmodelica.variable @v
    // CHECK-DAG: bmodelica.variable @lambda
    // CHECK-DAG: bmodelica.variable @g
    // CHECK-DAG: bmodelica.variable @L
    // CHECK-DAG: bmodelica.variable @der_x
    // CHECK-DAG: bmodelica.variable @der_der_x
    // CHECK-DAG: bmodelica.variable @der_y
    // CHECK-DAG: bmodelica.variable @der_der_y
    // CHECK-DAG: bmodelica.variable @der_u
    // CHECK-DAG: bmodelica.variable @der_v

    // COM: x' = u
    %t0 = bmodelica.equation_template inductions = [] {
      %0 = bmodelica.variable_get @der_x : f64
      %1 = bmodelica.variable_get @u : f64
      %2 = bmodelica.equation_side %0 : tuple<f64>
      %3 = bmodelica.equation_side %1 : tuple<f64>
      bmodelica.equation_sides %2, %3 : tuple<f64>, tuple<f64>
    }

    // CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[u:.*]] = bmodelica.variable_get @u
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[u]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: x'' = u'
    // CHECK:       %[[t0_der:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_der_x:.*]] = bmodelica.variable_get @der_der_x
    // CHECK-DAG:       %[[der_u:.*]] = bmodelica.variable_get @der_u
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_der_x]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[der_u]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: y' = v
    %t1 = bmodelica.equation_template inductions = [] {
      %0 = bmodelica.variable_get @der_y : f64
      %1 = bmodelica.variable_get @v : f64
      %2 = bmodelica.equation_side %0 : tuple<f64>
      %3 = bmodelica.equation_side %1 : tuple<f64>
      bmodelica.equation_sides %2, %3 : tuple<f64>, tuple<f64>
    }

    // CHECK:       %[[t1:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_y:.*]] = bmodelica.variable_get @der_y
    // CHECK-DAG:       %[[v:.*]] = bmodelica.variable_get @v
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_y]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[v]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: y'' = v'
    // CHECK:       %[[t1_der:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_der_y:.*]] = bmodelica.variable_get @der_der_y
    // CHECK-DAG:       %[[der_v:.*]] = bmodelica.variable_get @der_v
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_der_y]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[der_v]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: u' = lambda * x
    %t2 = bmodelica.equation_template inductions = [] {
      %0 = bmodelica.variable_get @der_u : f64
      %1 = bmodelica.variable_get @lambda : f64
      %2 = bmodelica.variable_get @x : f64
      %3 = bmodelica.mul %1, %2 : (f64, f64) -> f64
      %4 = bmodelica.equation_side %0 : tuple<f64>
      %5 = bmodelica.equation_side %3 : tuple<f64>
      bmodelica.equation_sides %4, %5 : tuple<f64>, tuple<f64>
    }

    // CHECK:       %[[t2:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_u:.*]] = bmodelica.variable_get @der_u
    // CHECK-DAG:       %[[lambda:.*]] = bmodelica.variable_get @lambda
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[mul:.*]] = bmodelica.mul %[[lambda]], %[[x]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_u]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[mul]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: v' = lambda * v - g
    %t3 = bmodelica.equation_template inductions = [] {
      %0 = bmodelica.variable_get @der_v : f64
      %1 = bmodelica.variable_get @lambda : f64
      %2 = bmodelica.variable_get @y : f64
      %3 = bmodelica.variable_get @g : f64
      %4 = bmodelica.mul %1, %2 : (f64, f64) -> f64
      %5 = bmodelica.sub %4, %3 : (f64, f64) -> f64
      %6 = bmodelica.equation_side %0 : tuple<f64>
      %7 = bmodelica.equation_side %5 : tuple<f64>
      bmodelica.equation_sides %6, %7 : tuple<f64>, tuple<f64>
    }

    // CHECK:       %[[t3:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[der_v:.*]] = bmodelica.variable_get @der_v
    // CHECK-DAG:       %[[lambda:.*]] = bmodelica.variable_get @lambda
    // CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[g:.*]] = bmodelica.variable_get @g
    // CHECK-DAG:       %[[mul:.*]] = bmodelica.mul %[[lambda]], %[[y]]
    // CHECK-DAG:       %[[sub:.*]] = bmodelica.sub %[[mul]], %[[g]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[der_v]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[sub]]
    // CHECK-DAG:       bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: x^2 + y^2 = L
    %t4 = bmodelica.equation_template inductions = [] {
      %0 = bmodelica.variable_get @x : f64
      %1 = bmodelica.variable_get @y : f64
      %2 = bmodelica.constant 2.0 : f64
      %3 = bmodelica.pow %0, %2 : (f64, f64) -> f64
      %4 = bmodelica.pow %1, %2 : (f64, f64) -> f64
      %5 = bmodelica.add %3, %4 : (f64, f64) -> f64
      %6 = bmodelica.variable_get @L : f64
      %7 = bmodelica.equation_side %5 : tuple<f64>
      %8 = bmodelica.equation_side %6 : tuple<f64>
      bmodelica.equation_sides %7, %8 : tuple<f64>, tuple<f64>
    }

    // CHECK:       %[[t4:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[const:.*]] = bmodelica.constant 2.0
    // CHECK-DAG:       %[[powX:.*]] = bmodelica.pow %[[x]], %[[const]]
    // CHECK-DAG:       %[[powY:.*]] = bmodelica.pow %[[y]], %[[const]]
    // CHECK-DAG:       %[[add:.*]] = bmodelica.add %[[powX]], %[[powY]]
    // CHECK-DAG:       %[[L:.*]] = bmodelica.variable_get @L
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[add]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[L]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM: 2 * x * x' + 2 * y * y' = 0
    // CHECK:       %[[t4_der:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK-DAG:       %[[zero:.*]] = bmodelica.constant 0.0
    // CHECK-DAG:       %[[two:.*]] = bmodelica.constant 2.0
    // CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
    // CHECK-DAG:       %[[der_x:.*]] = bmodelica.variable_get @der_x
    // CHECK-DAG:       %[[y:.*]] = bmodelica.variable_get @y
    // CHECK-DAG:       %[[der_y:.*]] = bmodelica.variable_get @der_y
    // CHECK-DAG:       %[[mul1:.*]] = bmodelica.mul %[[x]], %[[der_x]]
    // CHECK-DAG:       %[[mul2:.*]] = bmodelica.mul %[[mul1]], %[[two]]
    // CHECK-DAG:       %[[mul3:.*]] = bmodelica.mul %[[y]], %[[der_y]]
    // CHECK-DAG:       %[[mul4:.*]] = bmodelica.mul %[[mul3]], %[[two]]
    // CHECK-DAG:       %[[add:.*]] = bmodelica.add %[[mul2]], %[[mul4]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[add]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    // COM:
    // CHECK:       %[[t4_der_der:.*]] = bmodelica.equation_template inductions = [] {
    // CHECK:           %[[zero:.*]] = bmodelica.constant 0.0
    // CHECK:           %[[two:.*]] = bmodelica.constant 2.0
    // CHECK:           %[[x:.*]] = bmodelica.variable_get @x
    // CHECK:           %[[der_x_1:.*]] = bmodelica.variable_get @der_x
    // CHECK:           %[[der_x_2:.*]] = bmodelica.variable_get @der_x
    // CHECK:           %[[der_der_x:.*]] = bmodelica.variable_get @der_der_x
    // CHECK:           %[[y:.*]] = bmodelica.variable_get @y
    // CHECK:           %[[der_y_1:.*]] = bmodelica.variable_get @der_y
    // CHECK:           %[[der_y_2:.*]] = bmodelica.variable_get @der_y
    // CHECK:           %[[der_der_y:.*]] = bmodelica.variable_get @der_der_y
    // CHECK-DAG:       %[[mul1:.*]] = bmodelica.mul %[[der_x_1]], %[[der_x_2]]
    // CHECK-DAG:       %[[mul2:.*]] = bmodelica.mul %[[x]], %[[der_der_x]]
    // CHECK-DAG:       %[[add1:.*]] = bmodelica.add %[[mul1]], %[[mul2]]
    // CHECK-DAG:       %[[mul3:.*]] = bmodelica.mul %[[add1]], %[[two]]
    // CHECK-DAG:       %[[mul4:.*]] = bmodelica.mul %[[der_y_1]], %[[der_y_2]]
    // CHECK-DAG:       %[[mul5:.*]] = bmodelica.mul %[[y]], %[[der_der_y]]
    // CHECK-DAG:       %[[add2:.*]] = bmodelica.add %[[mul4]], %[[mul5]]
    // CHECK-DAG:       %[[mul6:.*]] = bmodelica.mul %[[add2]], %[[two]]
    // CHECK-DAG:       %[[add3:.*]] = bmodelica.add %[[mul3]], %[[mul6]]
    // CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[add3]]
    // CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[zero]]
    // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
    // CHECK-NEXT:  }

    bmodelica.dynamic {
      bmodelica.equation_instance %t0
      bmodelica.equation_instance %t1
      bmodelica.equation_instance %t2
      bmodelica.equation_instance %t3
      bmodelica.equation_instance %t4

      // CHECK-DAG: bmodelica.equation_instance %[[t0]]
      // CHECK-DAG: bmodelica.equation_instance %[[t0_der]]
      // CHECK-DAG: bmodelica.equation_instance %[[t1]]
      // CHECK-DAG: bmodelica.equation_instance %[[t1_der]]
      // CHECK-DAG: bmodelica.equation_instance %[[t2]]
      // CHECK-DAG: bmodelica.equation_instance %[[t3]]
      // CHECK-DAG: bmodelica.equation_instance %[[t4]]
      // CHECK-DAG: bmodelica.equation_instance %[[t4_der]]
      // CHECK-DAG: bmodelica.equation_instance %[[t4_der_der]]
    }
  }
}
