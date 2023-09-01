// RUN: modelica-opt %s --split-input-file --scalarize --canonicalize | FileCheck %s

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.abs %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_abs() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-2.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<3.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.abs %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.acos %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_acos() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.acos %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.asin %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_asin() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.asin %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.atan %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_atan() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.atan %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 4 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[value:.*]] = modelica.atan2 %{{.*}}, %{{.*}}
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_atan2() -> () {
    %y = modelica.alloca : !modelica.array<4x!modelica.real>
    %x = modelica.alloca : !modelica.array<4x!modelica.real>

    %c0 = modelica.constant 0 : index
    %y0 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    %x0 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    modelica.store %y[%c0], %y0 : !modelica.array<4x!modelica.real>
    modelica.store %x[%c0], %x0 : !modelica.array<4x!modelica.real>

    %c1 = modelica.constant 1 : index
    %y1 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    %x1 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    modelica.store %y[%c1], %y1 : !modelica.array<4x!modelica.real>
    modelica.store %x[%c1], %x1 : !modelica.array<4x!modelica.real>

    %c2 = modelica.constant 2 : index
    %y2 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    %x2 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    modelica.store %y[%c2], %y2 : !modelica.array<4x!modelica.real>
    modelica.store %x[%c2], %x2 : !modelica.array<4x!modelica.real>

    %c3 = modelica.constant 3 : index
    %y3 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    %x3 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    modelica.store %y[%c3], %y3 : !modelica.array<4x!modelica.real>
    modelica.store %x[%c3], %x3 : !modelica.array<4x!modelica.real>

    %result = modelica.atan2 %y, %x : (!modelica.array<4x!modelica.real>, !modelica.array<4x!modelica.real>) -> !modelica.array<4x!modelica.real>
    modelica.print %result : !modelica.array<4x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.ceil %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_ceil() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<-3.14> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<3.14> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.ceil %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.cos %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_cos() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<0.523598775> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<0.785398163> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.cos %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.cosh %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_cosh() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.cosh %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.exp %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_exp() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.exp %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.floor %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_floor() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<-3.14> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<3.14> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.floor %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.integer %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_integer() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<-3.14> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<3.14> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.integer %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.log %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_log() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<0.135335283> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<7.389056099> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.log %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.log10 %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_log10() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<0.01> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<100.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.log10 %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.sign %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_sign() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-2.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<3.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.sign %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.sin %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_sin() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<0.523598775> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<0.785398163> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.sin %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.sinh %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_sinh() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.sinh %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.sqrt %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_sqrt() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<4.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<9.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.sqrt %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.tan %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_tan() -> () {
    %array = modelica.alloca : !modelica.array<2x!modelica.real>

    %0 = modelica.constant #modelica.real<0.523598775> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<2x!modelica.real>

    %1 = modelica.constant #modelica.real<1.047197551> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<2x!modelica.real>

    %result = modelica.tan %array : !modelica.array<2x!modelica.real> -> !modelica.array<2x!modelica.real>
    modelica.print %result : !modelica.array<2x!modelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = modelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = modelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = modelica.tanh %[[load]]
// CHECK:           %[[subscription:.*]] = modelica.subscription %{{.*}}
// CHECK:           modelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.yield
// CHECK-NEXT:  }

func.func @test_tanh() -> () {
    %array = modelica.alloca : !modelica.array<3x!modelica.real>

    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    %c0 = modelica.constant 0 : index
    modelica.store %array[%c0], %0 : !modelica.array<3x!modelica.real>

    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    %c1 = modelica.constant 1 : index
    modelica.store %array[%c1], %1 : !modelica.array<3x!modelica.real>

    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    %c2 = modelica.constant 2 : index
    modelica.store %array[%c2], %2 : !modelica.array<3x!modelica.real>

    %result = modelica.tanh %array : !modelica.array<3x!modelica.real> -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>
    return
}
