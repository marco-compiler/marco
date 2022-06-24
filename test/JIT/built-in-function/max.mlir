// RUN: modelica-opt %s                             \
// RUN:     --convert-modelica-to-cf                \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: 2.500000e+00
// CHECK-NEXT: 2.500000e+00
// CHECK-NEXT: -1.500000e+00
// CHECK-NEXT: -1.500000e+00
// CHECK-NEXT: 1.500000e+00
// CHECK-NEXT: 1.500000e+00

func.func @test_scalars() -> () {
    %size = arith.constant 6 : index

    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %x = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %x0 = modelica.constant #modelica.real<1.5>
    %y0 = modelica.constant #modelica.real<2.5>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %x1 = modelica.constant #modelica.real<2.5>
    %y1 = modelica.constant #modelica.real<1.5>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %x2 = modelica.constant #modelica.real<-1.5>
    %y2 = modelica.constant #modelica.real<-2.5>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %x3 = modelica.constant #modelica.real<-2.5>
    %y3 = modelica.constant #modelica.real<-1.5>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>

    %c4 = arith.constant 4 : index
    %x4 = modelica.constant #modelica.real<1.5>
    %y4 = modelica.constant #modelica.real<-1.5>
    modelica.store %x[%c4], %x4 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c4], %y4 : !modelica.array<?x!modelica.real>

    %c5 = arith.constant 5 : index
    %x5 = modelica.constant #modelica.real<-1.5>
    %y5 = modelica.constant #modelica.real<1.5>
    modelica.store %x[%c5], %x5 : !modelica.array<?x!modelica.real>
    modelica.store %y[%c5], %y5 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.real>
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.max %xi, %yi : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT: 9.500000e+00

func.func @test_array() -> () {
    %size = arith.constant 6 : index
    %array = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %0 = modelica.constant #modelica.real<-1.5>
    modelica.store %array[%c0], %0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %1 = modelica.constant #modelica.real<9.5>
    modelica.store %array[%c1], %1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %2 = modelica.constant #modelica.real<-3.5>
    modelica.store %array[%c2], %2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %3 = modelica.constant #modelica.real<0.0>
    modelica.store %array[%c3], %3 : !modelica.array<?x!modelica.real>

    %c4 = arith.constant 4 : index
    %4 = modelica.constant #modelica.real<4.5>
    modelica.store %array[%c4], %4 : !modelica.array<?x!modelica.real>

    %c5 = arith.constant 5 : index
    %5 = modelica.constant #modelica.real<3.5>
    modelica.store %array[%c5], %5 : !modelica.array<?x!modelica.real>

    %result = modelica.max %array : !modelica.array<?x!modelica.real> -> !modelica.real
    modelica.print %result : !modelica.real

    return
}

func.func @main() -> () {
    call @test_scalars() : () -> ()
    call @test_array() : () -> ()
    return
}
