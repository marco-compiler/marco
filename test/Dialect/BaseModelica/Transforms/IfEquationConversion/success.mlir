// RUN: marco-opt %s --split-input-file --convert-if-equations | FileCheck %s

// COM: Tests that if_equation inside a bmodelica.initial block is converted to
// COM: a plain equation using bmodelica.select.

// CHECK-LABEL: bmodelica.model @InitialBlock
// CHECK-NOT:   bmodelica.if_equation
// CHECK:       bmodelica.initial {
// CHECK:           bmodelica.equation {
// CHECK-DAG:           %[[cond:.*]] = bmodelica.constant #bmodelica<bool true>
// CHECK-DAG:           %[[one:.*]]  = bmodelica.constant #bmodelica<int 1>
// CHECK-DAG:           %[[zero:.*]] = bmodelica.constant #bmodelica<int 0>
// CHECK-DAG:           %[[sel:.*]]  = bmodelica.select (%[[cond]] : !bmodelica.bool), (%[[one]] : !bmodelica.int), (%[[zero]] : !bmodelica.int)
// CHECK-DAG:           %[[x:.*]]    = bmodelica.variable.get @x
// CHECK-DAG:           %[[lhs:.*]]  = bmodelica.equation_side %[[x]]
// CHECK-DAG:           %[[rhs:.*]]  = bmodelica.equation_side %[[sel]]
// CHECK:               bmodelica.equation_sides %[[lhs]], %[[rhs]]

bmodelica.model @InitialBlock {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.initial {
        bmodelica.if_equation if {
            %cond = bmodelica.constant #bmodelica<bool true> : !bmodelica.bool
            bmodelica.yield %cond : !bmodelica.bool
        } then {
            bmodelica.equation {
                %x   = bmodelica.variable.get @x : !bmodelica.int
                %one = bmodelica.constant #bmodelica<int 1> : !bmodelica.int
                %lhs = bmodelica.equation_side %x   : tuple<!bmodelica.int>
                %rhs = bmodelica.equation_side %one : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        } else {
            bmodelica.equation {
                %x    = bmodelica.variable.get @x : !bmodelica.int
                %zero = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
                %lhs  = bmodelica.equation_side %x    : tuple<!bmodelica.int>
                %rhs  = bmodelica.equation_side %zero : tuple<!bmodelica.int>
                bmodelica.equation_sides %lhs, %rhs : tuple<!bmodelica.int>, tuple<!bmodelica.int>
            }
        }
    }
}
