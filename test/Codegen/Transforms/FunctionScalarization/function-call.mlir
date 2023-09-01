// RUN: modelica-opt %s --scalarize --canonicalize | FileCheck %s

// CHECK-LABEL: @caller
// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.call @callee(%[[load]]) : (!modelica.real) -> !modelica.real
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

modelica.function @callee {
    modelica.variable @x : !modelica.variable<!modelica.real, input>
    modelica.variable @y : !modelica.variable<!modelica.real, output>

    modelica.algorithm {
        %0 = modelica.variable_get @x : !modelica.real
        modelica.variable_set @y, %0 : !modelica.real
    }
}

func.func @caller() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %c0 = modelica.constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %c1 = modelica.constant 1 : index
    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %c2 = modelica.constant 2 : index
    %2 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.call @callee(%array) : (!modelica.array<3x!modelica.real>) -> (!modelica.array<3x!modelica.real>)
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}
