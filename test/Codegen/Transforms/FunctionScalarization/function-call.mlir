// RUN: modelica-opt %s --scalarize --canonicalize | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.call @callee(%[[load]]) : (!bmodelica.real) -> !bmodelica.real
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

bmodelica.function @callee {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real, input>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real, output>

    bmodelica.algorithm {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        bmodelica.variable_set @y, %0 : !bmodelica.real
    }
}

func.func @caller() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %c0 = bmodelica.constant 0 : index
    %0 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %c1 = bmodelica.constant 1 : index
    %1 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %c2 = bmodelica.constant 2 : index
    %2 = bmodelica.constant #bmodelica.real<2.0> : !bmodelica.real
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.call @callee(%array) : (!bmodelica.array<3x!bmodelica.real>) -> (!bmodelica.array<3x!bmodelica.real>)
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}
