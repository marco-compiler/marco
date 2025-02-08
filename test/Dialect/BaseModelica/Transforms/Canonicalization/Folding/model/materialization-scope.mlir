// RUN: modelica-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @equation

bmodelica.model @equation {
    bmodelica.dynamic {
        bmodelica.equation {
            %0 = bmodelica.constant #bmodelica<real 1.0>
            %1 = bmodelica.constant #bmodelica<real 4.0>
            %2 = bmodelica.constant #bmodelica<real 2.0>
            %3 = bmodelica.constant #bmodelica<real 3.0>
            %4 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            %5 = bmodelica.add %2, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            %6 = bmodelica.equation_side %4 : tuple<!bmodelica.real>
            %7 = bmodelica.equation_side %5 : tuple<!bmodelica.real>
            bmodelica.equation_sides %6, %7 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
        }

        // CHECK:       bmodelica.equation {
        // CHECK-NEXT:      %[[cst:.*]] = bmodelica.constant #bmodelica<real 5.000000e+00>
        // CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[cst]]
        // CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[cst]]
        // CHECK-NEXT:      bmodelica.equation_sides %[[lhs]], %[[rhs]]
        // CHECK-NEXT:  }
    }
}

// -----

// CHECK-LABEL: @algorithm

bmodelica.model @algorithm {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>

    bmodelica.dynamic {
        bmodelica.algorithm {
            %0 = bmodelica.constant #bmodelica<real 1.0>
            %1 = bmodelica.constant #bmodelica<real 2.0>
            %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
            bmodelica.variable_set @x, %2 : !bmodelica.real
        }

        // CHECK:       bmodelica.algorithm {
        // CHECK-NEXT:      %[[cst:.*]] = bmodelica.constant #bmodelica<real 3.000000e+00>
        // CHECK-NEXT:      bmodelica.variable_set @x, %[[cst]]
        // CHECK-NEXT:  }
    }
}
