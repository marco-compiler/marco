// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: 0
// CHECK-NEXT: 15
// CHECK-NEXT: -15
// CHECK-NEXT: -15

func @test_integerScalars() -> () {
    %size = constant 4 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %y = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<0> : !modelica.int
    %y0 = modelica.constant #modelica.int<10> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<3> : !modelica.int
    %y1 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %y2 = modelica.constant #modelica.int<-5> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.int>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<-5> : !modelica.int
    %y3 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %x[%c3], %x3 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, ?x!modelica.int>

    %lb = constant 0 : index
    %step = constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<stack, ?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.mul %xi, %yi : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 1.925000e+01
// CHECK-NEXT: -1.925000e+01
// CHECK-NEXT: -1.925000e+01

func @test_realScalars() -> () {
    %size = constant 4 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>
    %y = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.real<0.0> : !modelica.real
    %y0 = modelica.constant #modelica.real<10.5> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.real<3.5> : !modelica.real
    %y1 = modelica.constant #modelica.real<5.5> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.real<3.5> : !modelica.real
    %y2 = modelica.constant #modelica.real<-5.5> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.real<-3.5> : !modelica.real
    %y3 = modelica.constant #modelica.real<5.5> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<stack, ?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, ?x!modelica.real>

    %lb = constant 0 : index
    %step = constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<stack, ?x!modelica.real>
      %yi = modelica.load %y[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.mul %xi, %yi : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: 0.000000e+00
// CHECK-NEXT: 1.650000e+01
// CHECK-NEXT: -1.650000e+01
// CHECK-NEXT: -1.750000e+01

func @test_mixedScalars() -> () {
    %size = constant 4 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %y = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<0> : !modelica.int
    %y0 = modelica.constant #modelica.real<10.5> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<3> : !modelica.int
    %y1 = modelica.constant #modelica.real<5.5> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %y2 = modelica.constant #modelica.real<-5.5> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<-5> : !modelica.int
    %y3 = modelica.constant #modelica.real<3.5> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<stack, ?x!modelica.real>

    %lb = constant 0 : index
    %step = constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<stack, ?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.mul %xi, %yi : (!modelica.int, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: [10, -4, 0]

func @test_staticArrayAndScalar() -> () {
    %x = modelica.alloca : !modelica.array<stack, 3x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, 3x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 3x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 3x!modelica.int>

    %y = modelica.constant #modelica.int<2> : !modelica.int

    %result = modelica.mul %x, %y : (!modelica.array<stack, 3x!modelica.int>, !modelica.int) -> !modelica.array<stack, 3x!modelica.int>
    modelica.print %result : !modelica.array<stack, 3x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func @test_dynamicArrayAndScalar() -> () {
    %size = constant 3 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>

    %y = modelica.constant #modelica.int<2> : !modelica.int

    %result = modelica.mul %x, %y : (!modelica.array<stack, ?x!modelica.int>, !modelica.int) -> !modelica.array<stack, ?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func @test_scalarAndStaticArray() -> () {
    %x = modelica.alloca : !modelica.array<stack, 3x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, 3x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 3x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 3x!modelica.int>

    %y = modelica.constant #modelica.int<2> : !modelica.int

    %result = modelica.mul %y, %x : (!modelica.int, !modelica.array<stack, 3x!modelica.int>) -> !modelica.array<stack, 3x!modelica.int>
    modelica.print %result : !modelica.array<stack, 3x!modelica.int>

    return
}

// CHECK-NEXT: [10, -4, 0]

func @test_scalarAndDynamicArray() -> () {
    %size = constant 3 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<-2> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>

    %y = modelica.constant #modelica.int<2> : !modelica.int

    %result = modelica.mul %y, %x : (!modelica.int, !modelica.array<stack, ?x!modelica.int>) -> !modelica.array<stack, ?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x!modelica.int>

    return
}

// CHECK-NEXT: 32

func @test_static1dArrays() -> () {
    %x = modelica.alloca : !modelica.array<stack, 3x!modelica.int>
    %y = modelica.alloca : !modelica.array<stack, 3x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %y0 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, 3x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, 3x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<2> : !modelica.int
    %y1 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 3x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, 3x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %y2 = modelica.constant #modelica.int<6> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 3x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, 3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, 3x!modelica.int>, !modelica.array<stack, 3x!modelica.int>) -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT: 32

func @test_dynamic1dArrays() -> () {
    %size = constant 3 : index
    %x = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %y = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %y0 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<2> : !modelica.int
    %y1 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %y2 = modelica.constant #modelica.int<6> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, ?x!modelica.int>, !modelica.array<stack, ?x!modelica.int>) -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

func @test_static2dArrays() -> () {
    %x = modelica.alloca : !modelica.array<stack, 2x3x!modelica.int>
    %y = modelica.alloca : !modelica.array<stack, 3x2x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %x00 = modelica.constant #modelica.int<1> : !modelica.int
    %x01 = modelica.constant #modelica.int<2> : !modelica.int
    %x02 = modelica.constant #modelica.int<3> : !modelica.int
    %x10 = modelica.constant #modelica.int<4> : !modelica.int
    %x11 = modelica.constant #modelica.int<5> : !modelica.int
    %x12 = modelica.constant #modelica.int<6> : !modelica.int

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<stack, 2x3x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<stack, 2x3x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<stack, 2x3x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<stack, 2x3x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<stack, 2x3x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<stack, 2x3x!modelica.int>

    %y00 = modelica.constant #modelica.int<1> : !modelica.int
    %y01 = modelica.constant #modelica.int<2> : !modelica.int
    %y10 = modelica.constant #modelica.int<3> : !modelica.int
    %y11 = modelica.constant #modelica.int<4> : !modelica.int
    %y20 = modelica.constant #modelica.int<5> : !modelica.int
    %y21 = modelica.constant #modelica.int<6> : !modelica.int

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<stack, 3x2x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<stack, 3x2x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<stack, 3x2x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<stack, 3x2x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<stack, 3x2x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<stack, 3x2x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, 2x3x!modelica.int>, !modelica.array<stack, 3x2x!modelica.int>) -> !modelica.array<stack, 2x2x!modelica.int>
    modelica.print %result : !modelica.array<stack, 2x2x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [[22, 28], [49, 64]]

func @test_dynamic2dArrays() -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index

    %x = modelica.alloca %c2, %c3 : (index, index) -> !modelica.array<stack, ?x?x!modelica.int>
    %y = modelica.alloca %c3, %c2 : (index, index) -> !modelica.array<stack, ?x?x!modelica.int>

    %x00 = modelica.constant #modelica.int<1> : !modelica.int
    %x01 = modelica.constant #modelica.int<2> : !modelica.int
    %x02 = modelica.constant #modelica.int<3> : !modelica.int
    %x10 = modelica.constant #modelica.int<4> : !modelica.int
    %x11 = modelica.constant #modelica.int<5> : !modelica.int
    %x12 = modelica.constant #modelica.int<6> : !modelica.int

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<stack, ?x?x!modelica.int>

    %y00 = modelica.constant #modelica.int<1> : !modelica.int
    %y01 = modelica.constant #modelica.int<2> : !modelica.int
    %y10 = modelica.constant #modelica.int<3> : !modelica.int
    %y11 = modelica.constant #modelica.int<4> : !modelica.int
    %y20 = modelica.constant #modelica.int<5> : !modelica.int
    %y21 = modelica.constant #modelica.int<6> : !modelica.int

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<stack, ?x?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, ?x?x!modelica.int>, !modelica.array<stack, ?x?x!modelica.int>) -> !modelica.array<stack, ?x?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [70, 80, 90]

func @test_static1dArrayAndStatic2dArray() -> () {
    %x = modelica.alloca : !modelica.array<stack, 4x!modelica.int>
    %y = modelica.alloca : !modelica.array<stack, 4x3x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index

    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %x1 = modelica.constant #modelica.int<2> : !modelica.int
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %x3 = modelica.constant #modelica.int<4> : !modelica.int

    modelica.store %x[%c0], %x0 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %x[%c1], %x1 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %x[%c2], %x2 : !modelica.array<stack, 4x!modelica.int>
    modelica.store %x[%c3], %x3 : !modelica.array<stack, 4x!modelica.int>

    %y00 = modelica.constant #modelica.int<1> : !modelica.int
    %y01 = modelica.constant #modelica.int<2> : !modelica.int
    %y02 = modelica.constant #modelica.int<3> : !modelica.int
    %y10 = modelica.constant #modelica.int<4> : !modelica.int
    %y11 = modelica.constant #modelica.int<5> : !modelica.int
    %y12 = modelica.constant #modelica.int<6> : !modelica.int
    %y20 = modelica.constant #modelica.int<7> : !modelica.int
    %y21 = modelica.constant #modelica.int<8> : !modelica.int
    %y22 = modelica.constant #modelica.int<9> : !modelica.int
    %y30 = modelica.constant #modelica.int<10> : !modelica.int
    %y31 = modelica.constant #modelica.int<11> : !modelica.int
    %y32 = modelica.constant #modelica.int<12> : !modelica.int

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c0, %c2], %y02 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c1, %c2], %y12 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c2, %c2], %y22 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c3, %c0], %y30 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c3, %c1], %y31 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %y[%c3, %c2], %y32 : !modelica.array<stack, 4x3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, 4x!modelica.int>, !modelica.array<stack, 4x3x!modelica.int>) -> !modelica.array<stack, 3x!modelica.int>
    modelica.print %result : !modelica.array<stack, 3x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [70, 80, 90]

func @test_dynamic1dArrayAndDynamic2dArray() -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index

    %x = modelica.alloca %c4 : index -> !modelica.array<stack, ?x!modelica.int>
    %y = modelica.alloca %c4, %c3 : (index, index) -> !modelica.array<stack, ?x?x!modelica.int>

    %x0 = modelica.constant #modelica.int<1> : !modelica.int
    %x1 = modelica.constant #modelica.int<2> : !modelica.int
    %x2 = modelica.constant #modelica.int<3> : !modelica.int
    %x3 = modelica.constant #modelica.int<4> : !modelica.int

    modelica.store %x[%c0], %x0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %x[%c1], %x1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %x[%c2], %x2 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %x[%c3], %x3 : !modelica.array<stack, ?x!modelica.int>

    %y00 = modelica.constant #modelica.int<1> : !modelica.int
    %y01 = modelica.constant #modelica.int<2> : !modelica.int
    %y02 = modelica.constant #modelica.int<3> : !modelica.int
    %y10 = modelica.constant #modelica.int<4> : !modelica.int
    %y11 = modelica.constant #modelica.int<5> : !modelica.int
    %y12 = modelica.constant #modelica.int<6> : !modelica.int
    %y20 = modelica.constant #modelica.int<7> : !modelica.int
    %y21 = modelica.constant #modelica.int<8> : !modelica.int
    %y22 = modelica.constant #modelica.int<9> : !modelica.int
    %y30 = modelica.constant #modelica.int<10> : !modelica.int
    %y31 = modelica.constant #modelica.int<11> : !modelica.int
    %y32 = modelica.constant #modelica.int<12> : !modelica.int

    modelica.store %y[%c0, %c0], %y00 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c0, %c1], %y01 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c0, %c2], %y02 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c1, %c0], %y10 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c1, %c1], %y11 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c1, %c2], %y12 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c2, %c0], %y20 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c2, %c1], %y21 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c2, %c2], %y22 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c3, %c0], %y30 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c3, %c1], %y31 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %y[%c3, %c2], %y32 : !modelica.array<stack, ?x?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, ?x!modelica.int>, !modelica.array<stack, ?x?x!modelica.int>) -> !modelica.array<stack, ?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [14, 32, 50, 68]

func @test_static2dArrayAndStatic1dArray() -> () {
    %x = modelica.alloca : !modelica.array<stack, 4x3x!modelica.int>
    %y = modelica.alloca : !modelica.array<stack, 3x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index

    %x00 = modelica.constant #modelica.int<1> : !modelica.int
    %x01 = modelica.constant #modelica.int<2> : !modelica.int
    %x02 = modelica.constant #modelica.int<3> : !modelica.int
    %x10 = modelica.constant #modelica.int<4> : !modelica.int
    %x11 = modelica.constant #modelica.int<5> : !modelica.int
    %x12 = modelica.constant #modelica.int<6> : !modelica.int
    %x20 = modelica.constant #modelica.int<7> : !modelica.int
    %x21 = modelica.constant #modelica.int<8> : !modelica.int
    %x22 = modelica.constant #modelica.int<9> : !modelica.int
    %x30 = modelica.constant #modelica.int<10> : !modelica.int
    %x31 = modelica.constant #modelica.int<11> : !modelica.int
    %x32 = modelica.constant #modelica.int<12> : !modelica.int

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c2, %c0], %x20 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c2, %c1], %x21 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c2, %c2], %x22 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c3, %c0], %x30 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c3, %c1], %x31 : !modelica.array<stack, 4x3x!modelica.int>
    modelica.store %x[%c3, %c2], %x32 : !modelica.array<stack, 4x3x!modelica.int>

    %y0 = modelica.constant #modelica.int<1> : !modelica.int
    %y1 = modelica.constant #modelica.int<2> : !modelica.int
    %y2 = modelica.constant #modelica.int<3> : !modelica.int

    modelica.store %y[%c0], %y0 : !modelica.array<stack, 3x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, 3x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, 3x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, 4x3x!modelica.int>, !modelica.array<stack, 3x!modelica.int>) -> !modelica.array<stack, 4x!modelica.int>
    modelica.print %result : !modelica.array<stack, 4x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [14, 32, 50, 68]

func @test_dynamic2dArrayAndDynamic1dArray() -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index

    %x = modelica.alloca %c4, %c3 : (index, index) -> !modelica.array<stack, ?x?x!modelica.int>
    %y = modelica.alloca %c3 : index -> !modelica.array<stack, ?x!modelica.int>

    %x00 = modelica.constant #modelica.int<1> : !modelica.int
    %x01 = modelica.constant #modelica.int<2> : !modelica.int
    %x02 = modelica.constant #modelica.int<3> : !modelica.int
    %x10 = modelica.constant #modelica.int<4> : !modelica.int
    %x11 = modelica.constant #modelica.int<5> : !modelica.int
    %x12 = modelica.constant #modelica.int<6> : !modelica.int
    %x20 = modelica.constant #modelica.int<7> : !modelica.int
    %x21 = modelica.constant #modelica.int<8> : !modelica.int
    %x22 = modelica.constant #modelica.int<9> : !modelica.int
    %x30 = modelica.constant #modelica.int<10> : !modelica.int
    %x31 = modelica.constant #modelica.int<11> : !modelica.int
    %x32 = modelica.constant #modelica.int<12> : !modelica.int

    modelica.store %x[%c0, %c0], %x00 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c0, %c1], %x01 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c0, %c2], %x02 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c0], %x10 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c1], %x11 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c1, %c2], %x12 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c2, %c0], %x20 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c2, %c1], %x21 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c2, %c2], %x22 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c3, %c0], %x30 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c3, %c1], %x31 : !modelica.array<stack, ?x?x!modelica.int>
    modelica.store %x[%c3, %c2], %x32 : !modelica.array<stack, ?x?x!modelica.int>

    %y0 = modelica.constant #modelica.int<1> : !modelica.int
    %y1 = modelica.constant #modelica.int<2> : !modelica.int
    %y2 = modelica.constant #modelica.int<3> : !modelica.int

    modelica.store %y[%c0], %y0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<stack, ?x!modelica.int>

    %result = modelica.mul %x, %y : (!modelica.array<stack, ?x?x!modelica.int>, !modelica.array<stack, ?x!modelica.int>) -> !modelica.array<stack, ?x!modelica.int>
    modelica.print %result : !modelica.array<stack, ?x!modelica.int>

    return
}

func @main() -> () {
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
