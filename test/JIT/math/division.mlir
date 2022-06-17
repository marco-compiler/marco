// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 3
// CHECK-NEXT: -3
// CHECK-NEXT: -5

func.func @test_integerScalars() -> () {
    %size = arith.constant 4 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<0>
    %y0 = modelica.constant #modelica.int<10>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<15>
    %y1 = modelica.constant #modelica.int<5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<15>
    %y2 = modelica.constant #modelica.int<-5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-15>
    %y3 = modelica.constant #modelica.int<3>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.div %xi, %yi : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 2.818182e+00
// CHECK-NEXT: -2.818182e+00
// CHECK-NEXT: -2.818182e+00

func.func @test_realScalars() -> () {
    %size = arith.constant 4 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<0.0>
    %y0 = modelica.constant #modelica.real<10.5>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<15.5>
    %y1 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<15.5>
    %y2 = modelica.constant #modelica.real<-5.5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.real<-15.5>
    %y3 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.real>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.div %xi, %yi : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 2.727273e+00
// CHECK-NEXT: -2.727273e+00
// CHECK-NEXT: -4.285714e+00

func.func @test_mixedScalars() -> () {
    %size = arith.constant 4 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<0>
    %y0 = modelica.constant #modelica.real<10.5>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<15>
    %y1 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<15>
    %y2 = modelica.constant #modelica.real<-5.5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-15>
    %y3 = modelica.constant #modelica.real<3.5>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.div %xi, %yi : (!modelica.int, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: [5.000000e+00, -1.000000e+00, 0.000000e+00]

func.func @test_staticArrayAndScalar() -> () {
    %x = modelica.alloca : !modelica.array<3x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<10.0>
    modelica.store %x[%c0], %x0 : !modelica.array<3x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c1], %x1 : !modelica.array<3x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<0.0>
    modelica.store %x[%c2], %x2 : !modelica.array<3x!modelica.real>

    %y = modelica.constant #modelica.real<2.0>

    %result = modelica.div %x, %y : (!modelica.array<3x!modelica.real>, !modelica.real) -> !modelica.array<3x!modelica.real>
    modelica.print %result : !modelica.array<3x!modelica.real>

    return
}

// CHECK-NEXT{LITERAL}: [5.000000e+00, -1.000000e+00, 0.000000e+00]

func.func @test_dynamicArrayAndScalar() -> () {
    %size = arith.constant 3 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<10.0>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<-2.0>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<0.0>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>

    %y = modelica.constant #modelica.real<2.0>

    %result = modelica.div %x, %y : (!modelica.array<?x!modelica.real>, !modelica.real) -> !modelica.array<?x!modelica.real>
    modelica.print %result : !modelica.array<?x!modelica.real>

    return
}

func.func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()
    call @test_mixedScalars() : () -> ()

    call @test_staticArrayAndScalar() : () -> ()
    call @test_dynamicArrayAndScalar() : () -> ()

    return
}
