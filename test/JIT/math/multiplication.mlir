// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cf                \
// RUN:     --convert-modelica-to-arith             \
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
// CHECK-NEXT: 15
// CHECK-NEXT: -15
// CHECK-NEXT: -15

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
    %x1 = modelica.constant #modelica.int<3>
    %y1 = modelica.constant #modelica.int<5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<3>
    %y2 = modelica.constant #modelica.int<-5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-5>
    %y3 = modelica.constant #modelica.int<3>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.mul %xi, %yi : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 1.925000e+01
// CHECK-NEXT: -1.925000e+01
// CHECK-NEXT: -1.925000e+01

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
    %x1 = modelica.constant #modelica.real<3.5>
    %y1 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<3.5>
    %y2 = modelica.constant #modelica.real<-5.5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.real<-3.5>
    %y3 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.real>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.mul %xi, %yi : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 1.650000e+01
// CHECK-NEXT: -1.650000e+01
// CHECK-NEXT: -1.750000e+01

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
    %x1 = modelica.constant #modelica.int<3>
    %y1 = modelica.constant #modelica.real<5.5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<3>
    %y2 = modelica.constant #modelica.real<-5.5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.int<-5>
    %y3 = modelica.constant #modelica.real<3.5>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.mul %xi, %yi : (!modelica.int, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: [10, -4, 0]

func.func @test_staticArrayAndScalar() -> () {
    %x = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<5>
    modelica.store %x[%c0], %x0 : !modelica.array<3x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<3x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<0>
    modelica.store %x[%c2], %x2 : !modelica.array<3x!modelica.int>

    %y = modelica.constant #modelica.int<2>

    %result = modelica.mul %x, %y : (!modelica.array<3x!modelica.int>, !modelica.int) -> !modelica.array<3x!modelica.int>
    modelica.print %result : !modelica.array<3x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func.func @test_dynamicArrayAndScalar() -> () {
    %size = arith.constant 3 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<5>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<0>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>

    %y = modelica.constant #modelica.int<2>

    %result = modelica.mul %x, %y : (!modelica.array<?x!modelica.int>, !modelica.int) -> !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func.func @test_scalarAndStaticArray() -> () {
    %x = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<5>
    modelica.store %x[%c0], %x0 : !modelica.array<3x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<3x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<0>
    modelica.store %x[%c2], %x2 : !modelica.array<3x!modelica.int>

    %y = modelica.constant #modelica.int<2>

    %result = modelica.mul %y, %x : (!modelica.int, !modelica.array<3x!modelica.int>) -> !modelica.array<3x!modelica.int>
    modelica.print %result : !modelica.array<3x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func.func @test_scalarAndDynamicArray() -> () {
    %size = arith.constant 3 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<5>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<-2>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<0>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>

    %y = modelica.constant #modelica.int<2>

    %result = modelica.mul %y, %x : (!modelica.int, !modelica.array<?x!modelica.int>) -> !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>

    return
}

// CHECK-NEXT: 32

func.func @test_static1dArrays() -> () {
    %x = modelica.alloca : !modelica.array<3x!modelica.int>
    %y = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<1>
    %y0 = modelica.constant #modelica.int<4>
    modelica.store %x[%c0], %x0 : !modelica.array<3x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<3x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<2>
    %y1 = modelica.constant #modelica.int<5>
    modelica.store %x[%c1], %x1 : !modelica.array<3x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<3x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<3>
    %y2 = modelica.constant #modelica.int<6>
    modelica.store %x[%c2], %x2 : !modelica.array<3x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT: 32

func.func @test_dynamic1dArrays() -> () {
    %size = arith.constant 3 : index
    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.int<1>
    %y0 = modelica.constant #modelica.int<4>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.int<2>
    %y1 = modelica.constant #modelica.int<5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.int<3>
    %y2 = modelica.constant #modelica.int<6>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<?x!modelica.int>, !modelica.array<?x!modelica.int>) -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

func.func @test_static2dArrays() -> () {
    %x = modelica.alloca : !modelica.array<2x3x!modelica.int>
    %y = modelica.alloca : !modelica.array<3x2x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    %x00 = modelica.constant #modelica.int<1>
    %x01 = modelica.constant #modelica.int<2>
    %x02 = modelica.constant #modelica.int<3>
    %x10 = modelica.constant #modelica.int<4>
    %x11 = modelica.constant #modelica.int<5>
    %x12 = modelica.constant #modelica.int<6>

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<2x3x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<2x3x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<2x3x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<2x3x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<2x3x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<2x3x!modelica.int>

    %y00 = modelica.constant #modelica.int<1>
    %y01 = modelica.constant #modelica.int<2>
    %y10 = modelica.constant #modelica.int<3>
    %y11 = modelica.constant #modelica.int<4>
    %y20 = modelica.constant #modelica.int<5>
    %y21 = modelica.constant #modelica.int<6>

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<3x2x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<3x2x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<3x2x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<3x2x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<3x2x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<3x2x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<2x3x!modelica.int>, !modelica.array<3x2x!modelica.int>) -> !modelica.array<2x2x!modelica.int>
    modelica.print %result : !modelica.array<2x2x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

func.func @test_dynamic2dArrays() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %x = modelica.alloca %c2, %c3 : !modelica.array<?x?x!modelica.int>
    %y = modelica.alloca %c3, %c2 : !modelica.array<?x?x!modelica.int>

    %x00 = modelica.constant #modelica.int<1>
    %x01 = modelica.constant #modelica.int<2>
    %x02 = modelica.constant #modelica.int<3>
    %x10 = modelica.constant #modelica.int<4>
    %x11 = modelica.constant #modelica.int<5>
    %x12 = modelica.constant #modelica.int<6>

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<?x?x!modelica.int>

    %y00 = modelica.constant #modelica.int<1>
    %y01 = modelica.constant #modelica.int<2>
    %y10 = modelica.constant #modelica.int<3>
    %y11 = modelica.constant #modelica.int<4>
    %y20 = modelica.constant #modelica.int<5>
    %y21 = modelica.constant #modelica.int<6>

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<?x?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<?x?x!modelica.int>, !modelica.array<?x?x!modelica.int>) -> !modelica.array<?x?x!modelica.int>
    modelica.print %result : !modelica.array<?x?x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [70, 80, 90]

func.func @test_static1dArrayAndStatic2dArray() -> () {
    %x = modelica.alloca : !modelica.array<4x!modelica.int>
    %y = modelica.alloca : !modelica.array<4x3x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %x0 = modelica.constant #modelica.int<1>
    %x1 = modelica.constant #modelica.int<2>
    %x2 = modelica.constant #modelica.int<3>
    %x3 = modelica.constant #modelica.int<4>

    modelica.store %x[%c0], %x0 : !modelica.array<4x!modelica.int>
    modelica.store %x[%c1], %x1 : !modelica.array<4x!modelica.int>
    modelica.store %x[%c2], %x2 : !modelica.array<4x!modelica.int>
    modelica.store %x[%c3], %x3 : !modelica.array<4x!modelica.int>

    %y00 = modelica.constant #modelica.int<1>
    %y01 = modelica.constant #modelica.int<2>
    %y02 = modelica.constant #modelica.int<3>
    %y10 = modelica.constant #modelica.int<4>
    %y11 = modelica.constant #modelica.int<5>
    %y12 = modelica.constant #modelica.int<6>
    %y20 = modelica.constant #modelica.int<7>
    %y21 = modelica.constant #modelica.int<8>
    %y22 = modelica.constant #modelica.int<9>
    %y30 = modelica.constant #modelica.int<10>
    %y31 = modelica.constant #modelica.int<11>
    %y32 = modelica.constant #modelica.int<12>

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c0, %c2], %y02 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c1, %c2], %y12 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c2, %c2], %y22 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c3, %c0], %y30 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c3, %c1], %y31 : !modelica.array<4x3x!modelica.int>
    modelica.store %y[%c3, %c2], %y32 : !modelica.array<4x3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<4x!modelica.int>, !modelica.array<4x3x!modelica.int>) -> !modelica.array<3x!modelica.int>
    modelica.print %result : !modelica.array<3x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [70, 80, 90]

func.func @test_dynamic1dArrayAndDynamic2dArray() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %x = modelica.alloca %c4 : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %c4, %c3 : !modelica.array<?x?x!modelica.int>

    %x0 = modelica.constant #modelica.int<1>
    %x1 = modelica.constant #modelica.int<2>
    %x2 = modelica.constant #modelica.int<3>
    %x3 = modelica.constant #modelica.int<4>

    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>

    %y00 = modelica.constant #modelica.int<1>
    %y01 = modelica.constant #modelica.int<2>
    %y02 = modelica.constant #modelica.int<3>
    %y10 = modelica.constant #modelica.int<4>
    %y11 = modelica.constant #modelica.int<5>
    %y12 = modelica.constant #modelica.int<6>
    %y20 = modelica.constant #modelica.int<7>
    %y21 = modelica.constant #modelica.int<8>
    %y22 = modelica.constant #modelica.int<9>
    %y30 = modelica.constant #modelica.int<10>
    %y31 = modelica.constant #modelica.int<11>
    %y32 = modelica.constant #modelica.int<12>

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c0, %c2], %y02 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c1, %c2], %y12 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c2, %c2], %y22 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c3, %c0], %y30 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c3, %c1], %y31 : !modelica.array<?x?x!modelica.int>
    modelica.store %y[%c3, %c2], %y32 : !modelica.array<?x?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<?x!modelica.int>, !modelica.array<?x?x!modelica.int>) -> !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [14, 32, 50, 68]

func.func @test_static2dArrayAndStatic1dArray() -> () {
    %x = modelica.alloca : !modelica.array<4x3x!modelica.int>
    %y = modelica.alloca : !modelica.array<3x!modelica.int>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    %x00 = modelica.constant #modelica.int<1>
    %x01 = modelica.constant #modelica.int<2>
    %x02 = modelica.constant #modelica.int<3>
    %x10 = modelica.constant #modelica.int<4>
    %x11 = modelica.constant #modelica.int<5>
    %x12 = modelica.constant #modelica.int<6>
    %x20 = modelica.constant #modelica.int<7>
    %x21 = modelica.constant #modelica.int<8>
    %x22 = modelica.constant #modelica.int<9>
    %x30 = modelica.constant #modelica.int<10>
    %x31 = modelica.constant #modelica.int<11>
    %x32 = modelica.constant #modelica.int<12>

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c2, %c0], %x20 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c2, %c1], %x21 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c2, %c2], %x22 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c3, %c0], %x30 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c3, %c1], %x31 : !modelica.array<4x3x!modelica.int>
    modelica.store %x[%c3, %c2], %x32 : !modelica.array<4x3x!modelica.int>

    %y0 = modelica.constant #modelica.int<1>
    %y1 = modelica.constant #modelica.int<2>
    %y2 = modelica.constant #modelica.int<3>

    modelica.store %y[%c0], %y0 : !modelica.array<3x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<3x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<4x3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.array<4x!modelica.int>
    modelica.print %result : !modelica.array<4x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [14, 32, 50, 68]

func.func @test_dynamic2dArrayAndDynamic1dArray() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %x = modelica.alloca %c4, %c3 : !modelica.array<?x?x!modelica.int>
    %y = modelica.alloca %c3 : !modelica.array<?x!modelica.int>

    %x00 = modelica.constant #modelica.int<1>
    %x01 = modelica.constant #modelica.int<2>
    %x02 = modelica.constant #modelica.int<3>
    %x10 = modelica.constant #modelica.int<4>
    %x11 = modelica.constant #modelica.int<5>
    %x12 = modelica.constant #modelica.int<6>
    %x20 = modelica.constant #modelica.int<7>
    %x21 = modelica.constant #modelica.int<8>
    %x22 = modelica.constant #modelica.int<9>
    %x30 = modelica.constant #modelica.int<10>
    %x31 = modelica.constant #modelica.int<11>
    %x32 = modelica.constant #modelica.int<12>

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c2, %c0], %x20 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c2, %c1], %x21 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c2, %c2], %x22 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c3, %c0], %x30 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c3, %c1], %x31 : !modelica.array<?x?x!modelica.int>
    modelica.store %x[%c3, %c2], %x32 : !modelica.array<?x?x!modelica.int>

    %y0 = modelica.constant #modelica.int<1>
    %y1 = modelica.constant #modelica.int<2>
    %y2 = modelica.constant #modelica.int<3>

    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<?x?x!modelica.int>, !modelica.array<?x!modelica.int>) -> !modelica.array<?x!modelica.int>
    modelica.print %result : !modelica.array<?x!modelica.int>

    return
}

func.func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()
    call @test_mixedScalars() : () -> ()

    call @test_staticArrayAndScalar() : () -> ()
    call @test_dynamicArrayAndScalar() : () -> ()

    call @test_scalarAndStaticArray() : () -> ()
    call @test_scalarAndDynamicArray() : () -> ()

    call @test_static1dArrays() : () -> ()
    call @test_dynamic1dArrays() : () -> ()

    call @test_static2dArrays() : () -> ()
    call @test_dynamic2dArrays() : () -> ()

    call @test_static1dArrayAndStatic2dArray() : () -> ()
    call @test_dynamic1dArrayAndDynamic2dArray() : () -> ()

    call @test_static2dArrayAndStatic1dArray() : () -> ()
    call @test_dynamic2dArrayAndDynamic1dArray() : () -> ()

    return
}
