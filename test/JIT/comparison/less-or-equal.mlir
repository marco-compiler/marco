// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica                      \
// RUN:     --convert-modelica-to-cfg               \
// RUN:     --convert-to-llvm                       \
// RUN: | mlir-opt                                  \
// RUN:      --convert-scf-to-std                   \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: true

func @test_integerScalars() -> () {
    %size = constant 5 : index

    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<0> : !modelica.int
    %y0 = modelica.constant #modelica.int<0> : !modelica.int
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.int>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<10> : !modelica.int
    %y1 = modelica.constant #modelica.int<10> : !modelica.int
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.int>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<10> : !modelica.int
    %y2 = modelica.constant #modelica.int<11> : !modelica.int
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.int>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<10> : !modelica.int
    %y3 = modelica.constant #modelica.int<9> : !modelica.int
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.int>

    %c4 = constant 4 : index
    %x4 = modelica.constant #modelica.int<-11> : !modelica.int
    %y4 = modelica.constant #modelica.int<-10> : !modelica.int
    modelica.store %x[%c4], %x4 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c4], %y4 : !modelica.array<?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.lte %xi, %yi : (!modelica.int, !modelica.int) -> !modelica.bool
      modelica.print %result : !modelica.bool
    }

    return
}

// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: true

func @test_realScalars() -> () {
    %size = constant 5 : index

    %x = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.real<0.5> : !modelica.real
    %y0 = modelica.constant #modelica.real<0.5> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.real<10.5> : !modelica.real
    %y1 = modelica.constant #modelica.real<10.5> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.real<10.0> : !modelica.real
    %y2 = modelica.constant #modelica.real<10.5> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.real<10.0> : !modelica.real
    %y3 = modelica.constant #modelica.real<9.5> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %c4 = constant 4 : index
    %x4 = modelica.constant #modelica.real<-11.0> : !modelica.real
    %y4 = modelica.constant #modelica.real<-10.5> : !modelica.real
    modelica.store %x[%c4], %x4 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c4], %y4 : !modelica.array<?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.real>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.lte %xi, %yi : (!modelica.real, !modelica.real) -> !modelica.bool
      modelica.print %result : !modelica.bool
    }

    return
}

// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: true
// CHECK-NEXT: false
// CHECK-NEXT: true

func @test_mixedScalars() -> () {
    %size = constant 5 : index

    %x = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = constant 0 : index
    %x0 = modelica.constant #modelica.int<0> : !modelica.int
    %y0 = modelica.constant #modelica.real<0.5> : !modelica.real
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = constant 1 : index
    %x1 = modelica.constant #modelica.int<10> : !modelica.int
    %y1 = modelica.constant #modelica.real<10.5> : !modelica.real
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = constant 2 : index
    %x2 = modelica.constant #modelica.int<10> : !modelica.int
    %y2 = modelica.constant #modelica.real<10.0> : !modelica.real
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = constant 3 : index
    %x3 = modelica.constant #modelica.int<10> : !modelica.int
    %y3 = modelica.constant #modelica.real<9.5> : !modelica.real
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %c4 = constant 4 : index
    %x4 = modelica.constant #modelica.int<-10> : !modelica.int
    %y4 = modelica.constant #modelica.real<-9.5> : !modelica.real
    modelica.store %x[%c4], %x4 : !modelica.array<?x!modelica.int>
    modelica.store %y[%c4], %y4 : !modelica.array<?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.int>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.lte %xi, %yi : (!modelica.int, !modelica.real) -> !modelica.bool
      modelica.print %result : !modelica.bool
    }

    return
}

func @main() -> () {
    call @test_integerScalars() : () -> ()
    call @test_realScalars() : () -> ()
    call @test_mixedScalars() : () -> ()

    return
}
