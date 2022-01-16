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

// CHECK{LITERAL}: 1.500000e+00
// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 1.500000e+00

func @test_abs() -> () {
    %size = constant 3 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<-1.5> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<1.5> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.abs %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 5.235988e-01
// CHECK-NEXT{LITERAL}: 7.853982e-01
// CHECK-NEXT{LITERAL}: 1.570796e+00
// CHECK-NEXT{LITERAL}: 2.356194e+00
// CHECK-NEXT{LITERAL}: 2.617994e+00
// CHECK-NEXT{LITERAL}: 3.141593e+00

func @test_acos() -> () {
    %size = constant 7 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.866025403> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    modelica.store %array[%c4], %4 : !modelica.array<stack, ?x!modelica.real>

    %c5 = constant 5 : index
    %5 = modelica.constant #modelica.real<-0.866025403> : !modelica.real
    modelica.store %array[%c5], %5 : !modelica.array<stack, ?x!modelica.real>

    %c6 = constant 6 : index
    %6 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c6], %6 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.acos %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 1.570796e+00
// CHECK-NEXT{LITERAL}: 1.047198e+00
// CHECK-NEXT{LITERAL}: 7.853982e-01
// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: -7.853982e-01
// CHECK-NEXT{LITERAL}: -1.047198e+00
// CHECK-NEXT{LITERAL}: -1.570796e+00

func @test_asin() -> () {
    %size = constant 7 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.866025403> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.707106781> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.real<-0.707106781> : !modelica.real
    modelica.store %array[%c4], %4 : !modelica.array<stack, ?x!modelica.real>

    %c5 = constant 5 : index
    %5 = modelica.constant #modelica.real<-0.866025403> : !modelica.real
    modelica.store %array[%c5], %5 : !modelica.array<stack, ?x!modelica.real>

    %c6 = constant 6 : index
    %6 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c6], %6 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.asin %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 7.853982e-01
// CHECK-NEXT{LITERAL}: 5.235988e-01
// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: -5.235988e-01
// CHECK-NEXT{LITERAL}: -7.853982e-01

func @test_atan() -> () {
    %size = constant 5 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.577350269> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<-0.577350269> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c4], %4 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.atan %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 8.660254e-01
// CHECK-NEXT{LITERAL}: 7.071068e-01

func @test_cos() -> () {
    %size = constant 3 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.523598775> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.785398163> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.cos %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 1.543081e+00
// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 1.543081e+00

func @test_cosh() -> () {
    %size = constant 3 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.cosh %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]

func @test_diagonal() -> () {
    %diagonal = modelica.alloca : !modelica.array<stack, 3x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %diagonal[%c0], %0 : !modelica.array<stack, 3x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %diagonal[%c1], %1 : !modelica.array<stack, 3x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %diagonal[%c2], %2 : !modelica.array<stack, 3x!modelica.int>

    %result = modelica.diagonal %diagonal : !modelica.array<stack, 3x!modelica.int> -> !modelica.array<stack, 3x3x!modelica.int>
    modelica.print %result : !modelica.array<stack, 3x3x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 2.718282e+00
// CHECK-NEXT{LITERAL}: 7.389056e+00
// CHECK-NEXT{LITERAL}: 1.353353e-01

func @test_exp() -> () {
    %size = constant 4 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<2.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<-2.0> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.exp %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: [[1]]
// CHECK-NEXT{LITERAL}: [[1, 0], [0, 1]]
// CHECK-NEXT{LITERAL}: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

func @test_identity() -> () {
    %size = constant 3 : index
    %dimensions = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %dimensions[%c0], %0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %dimensions[%c1], %1 : !modelica.array<stack, ?x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %dimensions[%c2], %2 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %dimension = modelica.load %dimensions[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.identity %dimension : !modelica.int -> !modelica.array<stack, ?x?x!modelica.int>
      modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>
    }

    return
}

// CHECK-NEXT{LITERAL}: [-1.000000e+00, -9.000000e-01, -8.000000e-01, -7.000000e-01, -6.000000e-01, -5.000000e-01, -4.000000e-01, -3.000000e-01, -2.000000e-01, -1.000000e-01, 0.000000e+00]
// CHECK-NEXT{LITERAL}: [0.000000e+00, 1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01, 5.000000e-01, 6.000000e-01, 7.000000e-01, 8.000000e-01, 9.000000e-01, 1.000000e+00]

func @test_linspace() -> () {
    %size = constant 2 : index

    %begins = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %ends = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %stepsCounts = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %begin0 = modelica.constant #modelica.int<-1> : !modelica.int
    %end0 = modelica.constant #modelica.int<0> : !modelica.int
    %stepsCount0 = modelica.constant #modelica.int<11> : !modelica.int
    modelica.store %begins[%c0], %begin0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %ends[%c0], %end0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %stepsCounts[%c0], %stepsCount0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %begin1 = modelica.constant #modelica.int<0> : !modelica.int
    %end1 = modelica.constant #modelica.int<1> : !modelica.int
    %stepsCount1 = modelica.constant #modelica.int<11> : !modelica.int
    modelica.store %begins[%c1], %begin1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %ends[%c1], %end1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %stepsCounts[%c1], %stepsCount1 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %begin = modelica.load %begins[%i] : !modelica.array<stack, ?x!modelica.int>
      %end = modelica.load %ends[%i] : !modelica.array<stack, ?x!modelica.int>
      %stepsCount = modelica.load %stepsCounts[%i] : !modelica.array<stack, ?x!modelica.int>

      %result = modelica.linspace %begin, %end, %stepsCount : (!modelica.int, !modelica.int, !modelica.int) -> !modelica.array<stack, ?x!modelica.real>
      modelica.print %result : !modelica.array<stack, ?x!modelica.real>
    }

    return
}

// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 2.000000e+00
// CHECK-NEXT{LITERAL}: -1.000000e+00

func @test_log() -> () {
    %size = constant 4 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<2.718281828> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<7.389056099> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<0.367879441> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.log %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 2.000000e+00
// CHECK-NEXT{LITERAL}: -1.000000e+00

func @test_log10() -> () {
    %size = constant 4 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<10.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<100.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<0.1> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.log10 %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: [[1, 1, 1], [1, 1, 1]]
// CHECK-NEXT{LITERAL}: [[1, 1], [1, 1], [1, 1]]

func @test_ones() -> () {
    %size = constant 2 : index

    %firstDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %secondDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %firstDim0 = modelica.constant #modelica.int<2> : !modelica.int
    %secondDim0 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %firstDims[%c0], %firstDim0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c0], %secondDim0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %firstDim1 = modelica.constant #modelica.int<3> : !modelica.int
    %secondDim1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %firstDims[%c1], %firstDim1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c1], %secondDim1 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %firstDim = modelica.load %firstDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %secondDim = modelica.load %secondDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.ones %firstDim, %secondDim : (!modelica.int, !modelica.int) -> !modelica.array<stack, ?x?x!modelica.int>
      modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>
    }

    return
}

// CHECK-NEXT{LITERAL}: 120

func @test_product() -> () {
    %array = modelica.alloca : !modelica.array<stack, 5x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %array[%c0], %0 : !modelica.array<stack, 5x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %array[%c1], %1 : !modelica.array<stack, 5x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %array[%c2], %2 : !modelica.array<stack, 5x!modelica.int>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %array[%c3], %3 : !modelica.array<stack, 5x!modelica.int>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %array[%c4], %4 : !modelica.array<stack, 5x!modelica.int>

    %result = modelica.product %array : !modelica.array<stack, 5x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 5.000000e-01
// CHECK-NEXT{LITERAL}: 7.071068e-01

func @test_sin() -> () {
    %size = constant 3 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.523598775> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<0.785398163> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.sin %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: -1.175201e+00
// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 1.175201e+00

func @test_sinh() -> () {
    %size = constant 3 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<-1.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.sinh %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 0.000000e+00
// CHECK-NEXT{LITERAL}: 1.000000e+00
// CHECK-NEXT{LITERAL}: 2.000000e+00
// CHECK-NEXT{LITERAL}: 3.000000e+00

func @test_sqrt() -> () {
    %size = constant 4 : index
    %array = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.real>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.real<0.0> : !modelica.real
    modelica.store %array[%c0], %0 : !modelica.array<stack, ?x!modelica.real>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.real<1.0> : !modelica.real
    modelica.store %array[%c1], %1 : !modelica.array<stack, ?x!modelica.real>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.real<4.0> : !modelica.real
    modelica.store %array[%c2], %2 : !modelica.array<stack, ?x!modelica.real>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.real<9.0> : !modelica.real
    modelica.store %array[%c3], %3 : !modelica.array<stack, ?x!modelica.real>

    scf.for %i = %c0 to %size step %c1 {
      %value = modelica.load %array[%i] : !modelica.array<stack, ?x!modelica.real>
      %result = modelica.sqrt %value : !modelica.real -> !modelica.real
      modelica.print %result : !modelica.real
    }

    return
}

// CHECK-NEXT{LITERAL}: 15

func @test_sum() -> () {
    %array = modelica.alloca : !modelica.array<stack, 5x!modelica.int>

    %c0 = constant 0 : index
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %array[%c0], %0 : !modelica.array<stack, 5x!modelica.int>

    %c1 = constant 1 : index
    %1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %array[%c1], %1 : !modelica.array<stack, 5x!modelica.int>

    %c2 = constant 2 : index
    %2 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %array[%c2], %2 : !modelica.array<stack, 5x!modelica.int>

    %c3 = constant 3 : index
    %3 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %array[%c3], %3 : !modelica.array<stack, 5x!modelica.int>

    %c4 = constant 4 : index
    %4 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %array[%c4], %4 : !modelica.array<stack, 5x!modelica.int>

    %result = modelica.sum %array : !modelica.array<stack, 5x!modelica.int> -> !modelica.int
    modelica.print %result : !modelica.int

    return
}

// CHECK-NEXT{LITERAL}: [[1, 3, 5], [2, 4, 6]]

func @test_transpose() -> () {
    %matrix = modelica.alloca : !modelica.array<stack, 3x2x!modelica.int>

    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index

    %00 = modelica.constant #modelica.int<1> : !modelica.int
    modelica.store %matrix[%c0, %c0], %00 : !modelica.array<stack, 3x2x!modelica.int>

    %01 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %matrix[%c0, %c1], %01 : !modelica.array<stack, 3x2x!modelica.int>

    %10 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %matrix[%c1, %c0], %10 : !modelica.array<stack, 3x2x!modelica.int>

    %11 = modelica.constant #modelica.int<4> : !modelica.int
    modelica.store %matrix[%c1, %c1], %11 : !modelica.array<stack, 3x2x!modelica.int>

    %20 = modelica.constant #modelica.int<5> : !modelica.int
    modelica.store %matrix[%c2, %c0], %20 : !modelica.array<stack, 3x2x!modelica.int>

    %21 = modelica.constant #modelica.int<6> : !modelica.int
    modelica.store %matrix[%c2, %c1], %21 : !modelica.array<stack, 3x2x!modelica.int>

    %result = modelica.transpose %matrix : !modelica.array<stack, 3x2x!modelica.int> -> !modelica.array<stack, 2x3x!modelica.int>
    modelica.print %result : !modelica.array<stack, 2x3x!modelica.int>

    return
}

// CHECK-NEXT{LITERAL}: [[0, 0, 0], [0, 0, 0]]
// CHECK-NEXT{LITERAL}: [[0, 0], [0, 0], [0, 0]]

func @test_zeros() -> () {
    %size = constant 2 : index

    %firstDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>
    %secondDims = modelica.alloca %size : index -> !modelica.array<stack, ?x!modelica.int>

    %c0 = constant 0 : index
    %firstDim0 = modelica.constant #modelica.int<2> : !modelica.int
    %secondDim0 = modelica.constant #modelica.int<3> : !modelica.int
    modelica.store %firstDims[%c0], %firstDim0 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c0], %secondDim0 : !modelica.array<stack, ?x!modelica.int>

    %c1 = constant 1 : index
    %firstDim1 = modelica.constant #modelica.int<3> : !modelica.int
    %secondDim1 = modelica.constant #modelica.int<2> : !modelica.int
    modelica.store %firstDims[%c1], %firstDim1 : !modelica.array<stack, ?x!modelica.int>
    modelica.store %secondDims[%c1], %secondDim1 : !modelica.array<stack, ?x!modelica.int>

    scf.for %i = %c0 to %size step %c1 {
      %firstDim = modelica.load %firstDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %secondDim = modelica.load %secondDims[%i] : !modelica.array<stack, ?x!modelica.int>
      %result = modelica.zeros %firstDim, %secondDim : (!modelica.int, !modelica.int) -> !modelica.array<stack, ?x?x!modelica.int>
      modelica.print %result : !modelica.array<stack, ?x?x!modelica.int>
    }

    return
}

func @main() -> () {
    call @test_abs() : () -> ()
    call @test_acos() : () -> ()
    call @test_asin() : () -> ()
    call @test_atan() : () -> ()
    call @test_cos() : () -> ()
    call @test_cosh() : () -> ()
    call @test_diagonal() : () -> ()
    call @test_exp() : () -> ()
    call @test_identity() : () -> ()
    call @test_linspace() : () -> ()
    call @test_log() : () -> ()
    call @test_log10() : () -> ()
    call @test_ones() : () -> ()
    call @test_product() : () -> ()
    call @test_sin() : () -> ()
    call @test_sinh() : () -> ()
    call @test_sqrt() : () -> ()
    call @test_sum() : () -> ()
    call @test_transpose() : () -> ()
    call @test_zeros() : () -> ()

    return
}
