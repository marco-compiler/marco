// RUN: modelica-opt %s --split-input-file --scalarize --canonicalize | FileCheck %s

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.abs %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_abs() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-2.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<3.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.abs %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.acos %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_acos() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.acos %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.asin %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_asin() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.asin %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.atan %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_atan() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.atan %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 4 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[value:.*]] = bmodelica.atan2 %{{.*}}, %{{.*}}
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_atan2() -> () {
    %y = bmodelica.alloca : <4x!bmodelica.real>
    %x = bmodelica.alloca : <4x!bmodelica.real>

    %c0 = bmodelica.constant 0 : index
    %y0 = bmodelica.constant #bmodelica.real<0.707106781> : !bmodelica.real
    %x0 = bmodelica.constant #bmodelica.real<0.707106781> : !bmodelica.real
    bmodelica.store %y[%c0], %y0 : !bmodelica.array<4x!bmodelica.real>
    bmodelica.store %x[%c0], %x0 : !bmodelica.array<4x!bmodelica.real>

    %c1 = bmodelica.constant 1 : index
    %y1 = bmodelica.constant #bmodelica.real<0.707106781> : !bmodelica.real
    %x1 = bmodelica.constant #bmodelica.real<-0.707106781> : !bmodelica.real
    bmodelica.store %y[%c1], %y1 : !bmodelica.array<4x!bmodelica.real>
    bmodelica.store %x[%c1], %x1 : !bmodelica.array<4x!bmodelica.real>

    %c2 = bmodelica.constant 2 : index
    %y2 = bmodelica.constant #bmodelica.real<-0.707106781> : !bmodelica.real
    %x2 = bmodelica.constant #bmodelica.real<-0.707106781> : !bmodelica.real
    bmodelica.store %y[%c2], %y2 : !bmodelica.array<4x!bmodelica.real>
    bmodelica.store %x[%c2], %x2 : !bmodelica.array<4x!bmodelica.real>

    %c3 = bmodelica.constant 3 : index
    %y3 = bmodelica.constant #bmodelica.real<-0.707106781> : !bmodelica.real
    %x3 = bmodelica.constant #bmodelica.real<0.707106781> : !bmodelica.real
    bmodelica.store %y[%c3], %y3 : !bmodelica.array<4x!bmodelica.real>
    bmodelica.store %x[%c3], %x3 : !bmodelica.array<4x!bmodelica.real>

    %result = bmodelica.atan2 %y, %x : (!bmodelica.array<4x!bmodelica.real>, !bmodelica.array<4x!bmodelica.real>) -> !bmodelica.array<4x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<4x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.ceil %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_ceil() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-3.14> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<3.14> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.ceil %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.cos %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_cos() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<0.523598775> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.785398163> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.cos %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.cosh %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_cosh() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.cosh %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.exp %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_exp() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.exp %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.floor %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_floor() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-3.14> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<3.14> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.floor %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.integer %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_integer() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-3.14> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<3.14> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.integer %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.log %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_log() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<0.135335283> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<7.389056099> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.log %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.log10 %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_log10() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<0.01> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<100.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.log10 %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.sign %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_sign() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-2.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<3.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.sign %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.sin %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_sin() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<0.523598775> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.785398163> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.sin %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.sinh %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_sinh() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.sinh %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.sqrt %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_sqrt() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<4.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<9.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.sqrt %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 2 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.tan %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_tan() -> () {
    %array = bmodelica.alloca : <2x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<0.523598775> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<2x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<1.047197551> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<2x!bmodelica.real>

    %result = bmodelica.tan %array : !bmodelica.array<2x!bmodelica.real> -> !bmodelica.array<2x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<2x!bmodelica.real>
    return
}

// -----

// CHECK-DAG:   %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[ub:.*]] = bmodelica.constant 3 : index
// CHECK-DAG:   %[[step:.*]] = arith.constant 1 : index
// CHECK:       scf.parallel (%[[i0:.*]]) = (%[[lb]]) to (%[[ub]]) step (%[[step]]) {
// CHECK:           %[[load:.*]] = bmodelica.load %{{.*}}[%[[i0]]]
// CHECK:           %[[value:.*]] = bmodelica.tanh %[[load]]
// CHECK:           %[[subscription:.*]] = bmodelica.subscription %{{.*}}
// CHECK:           bmodelica.assignment %[[subscription]], %[[value]]
// CHECK:           scf.reduce
// CHECK-NEXT:  }

func.func @test_tanh() -> () {
    %array = bmodelica.alloca : <3x!bmodelica.real>

    %0 = bmodelica.constant #bmodelica.real<-1.0> : !bmodelica.real
    %c0 = bmodelica.constant 0 : index
    bmodelica.store %array[%c0], %0 : !bmodelica.array<3x!bmodelica.real>

    %1 = bmodelica.constant #bmodelica.real<0.0> : !bmodelica.real
    %c1 = bmodelica.constant 1 : index
    bmodelica.store %array[%c1], %1 : !bmodelica.array<3x!bmodelica.real>

    %2 = bmodelica.constant #bmodelica.real<1.0> : !bmodelica.real
    %c2 = bmodelica.constant 2 : index
    bmodelica.store %array[%c2], %2 : !bmodelica.array<3x!bmodelica.real>

    %result = bmodelica.tanh %array : !bmodelica.array<3x!bmodelica.real> -> !bmodelica.array<3x!bmodelica.real>
    bmodelica.print %result : !bmodelica.array<3x!bmodelica.real>
    return
}
