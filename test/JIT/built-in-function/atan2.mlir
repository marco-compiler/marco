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

// CHECK: 7.853982e-01
// CHECK-NEXT: 2.356194e+00
// CHECK-NEXT: -2.356194e+00
// CHECK-NEXT: -7.853982e-01

func.func @test() -> () {
    %size = arith.constant 4 : index

    %y = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %x = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %y0 = modelica.constant #modelica.real<0.707106781>
    %x0 = modelica.constant #modelica.real<0.707106781>
    modelica.store %y[%c0], %y0 : !modelica.array<?x!modelica.real>
    modelica.store %x[%c0], %x0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %y1 = modelica.constant #modelica.real<0.707106781>
    %x1 = modelica.constant #modelica.real<-0.707106781>
    modelica.store %y[%c1], %y1 : !modelica.array<?x!modelica.real>
    modelica.store %x[%c1], %x1 : !modelica.array<?x!modelica.real>

    %c2 = arith.constant 2 : index
    %y2 = modelica.constant #modelica.real<-0.707106781>
    %x2 = modelica.constant #modelica.real<-0.707106781>
    modelica.store %y[%c2], %y2 : !modelica.array<?x!modelica.real>
    modelica.store %x[%c2], %x2 : !modelica.array<?x!modelica.real>

    %c3 = arith.constant 3 : index
    %y3 = modelica.constant #modelica.real<-0.707106781>
    %x3 = modelica.constant #modelica.real<0.707106781>
    modelica.store %y[%c3], %y3 : !modelica.array<?x!modelica.real>
    modelica.store %x[%c3], %x3 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %c0 to %size step %c1 {
      %yi = modelica.load %y[%i] : !modelica.array<?x!modelica.real>
      %xi = modelica.load %x[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.atan2 %yi, %xi : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

func.func @main() -> () {
    call @test() : () -> ()
    return
}
