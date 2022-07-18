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

// CHECK: 0
// CHECK-NEXT: 2

func.func @test_integers() -> () {
    %size = arith.constant 2 : index

    %dividends = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %divisors = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %dividend0 = modelica.constant #modelica.int<6>
    %divisor0 = modelica.constant #modelica.int<3>
    modelica.store %dividends[%c0], %dividend0 : !modelica.array<?x!modelica.int>
    modelica.store %divisors[%c0], %divisor0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %dividend1 = modelica.constant #modelica.int<8>
    %divisor1 = modelica.constant #modelica.int<3>
    modelica.store %dividends[%c1], %dividend1 : !modelica.array<?x!modelica.int>
    modelica.store %divisors[%c1], %divisor1 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %dividend = modelica.load %dividends[%i] : !modelica.array<?x!modelica.int>
      %divisor = modelica.load %divisors[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.mod %dividend, %divisor : (!modelica.int, !modelica.int) -> !modelica.int
      modelica.print %result : !modelica.int
    }

    return
}

// CHECK: 0.000000e+00
// CHECK-NEXT: 2.500000e+00

func.func @test_reals() -> () {
    %size = arith.constant 2 : index

    %dividends = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %divisors = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %dividend0 = modelica.constant #modelica.real<6.0>
    %divisor0 = modelica.constant #modelica.real<3.0>
    modelica.store %dividends[%c0], %dividend0 : !modelica.array<?x!modelica.real>
    modelica.store %divisors[%c0], %divisor0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %dividend1 = modelica.constant #modelica.real<8.5>
    %divisor1 = modelica.constant #modelica.real<3.0>
    modelica.store %dividends[%c1], %dividend1 : !modelica.array<?x!modelica.real>
    modelica.store %divisors[%c1], %divisor1 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %dividend = modelica.load %dividends[%i] : !modelica.array<?x!modelica.real>
      %divisor = modelica.load %divisors[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.mod %dividend, %divisor : (!modelica.real, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK: 0.000000e+00
// CHECK-NEXT: 2.500000e+00

func.func @test_realInteger() -> () {
    %size = arith.constant 2 : index

    %dividends = modelica.alloca %size : !modelica.array<?x!modelica.real>
    %divisors = modelica.alloca %size : !modelica.array<?x!modelica.int>

    %c0 = arith.constant 0 : index
    %dividend0 = modelica.constant #modelica.real<6.0>
    %divisor0 = modelica.constant #modelica.int<3>
    modelica.store %dividends[%c0], %dividend0 : !modelica.array<?x!modelica.real>
    modelica.store %divisors[%c0], %divisor0 : !modelica.array<?x!modelica.int>

    %c1 = arith.constant 1 : index
    %dividend1 = modelica.constant #modelica.real<8.5>
    %divisor1 = modelica.constant #modelica.int<3>
    modelica.store %dividends[%c1], %dividend1 : !modelica.array<?x!modelica.real>
    modelica.store %divisors[%c1], %divisor1 : !modelica.array<?x!modelica.int>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %dividend = modelica.load %dividends[%i] : !modelica.array<?x!modelica.real>
      %divisor = modelica.load %divisors[%i] : !modelica.array<?x!modelica.int>
      %result = modelica.mod %dividend, %divisor : (!modelica.real, !modelica.int) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK: 0.000000e+00
// CHECK-NEXT: 2.500000e+00

func.func @test_integerReal() -> () {
    %size = arith.constant 2 : index

    %dividends = modelica.alloca %size : !modelica.array<?x!modelica.int>
    %divisors = modelica.alloca %size : !modelica.array<?x!modelica.real>

    %c0 = arith.constant 0 : index
    %dividend0 = modelica.constant #modelica.int<6>
    %divisor0 = modelica.constant #modelica.real<3.0>
    modelica.store %dividends[%c0], %dividend0 : !modelica.array<?x!modelica.int>
    modelica.store %divisors[%c0], %divisor0 : !modelica.array<?x!modelica.real>

    %c1 = arith.constant 1 : index
    %dividend1 = modelica.constant #modelica.int<10>
    %divisor1 = modelica.constant #modelica.real<3.75>
    modelica.store %dividends[%c1], %dividend1 : !modelica.array<?x!modelica.int>
    modelica.store %divisors[%c1], %divisor1 : !modelica.array<?x!modelica.real>

    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    scf.for %i = %lb to %size step %step {
      %dividend = modelica.load %dividends[%i] : !modelica.array<?x!modelica.int>
      %divisor = modelica.load %divisors[%i] : !modelica.array<?x!modelica.real>
      %result = modelica.mod %dividend, %divisor : (!modelica.int, !modelica.real) -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

func.func @main() -> () {
    call @test_integers() : () -> ()
    call @test_reals() : () -> ()
    call @test_realInteger() : () -> ()
    call @test_integerReal() : () -> ()
    return
}
