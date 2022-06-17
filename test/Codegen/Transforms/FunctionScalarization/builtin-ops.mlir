// RUN: modelica-opt %s                             \
// RUN:     --scalarize                             \
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

// CHECK{LITERAL}: [2.000000e+00, 0.000000e+00, 3.000000e+00]

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

// CHECK{LITERAL}: [3.141593e+00, 1.570796e+00, 0.000000e+00]

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

// CHECK{LITERAL}: [-1.570796e+00, 0.000000e+00, 1.570796e+00]

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

// CHECK{LITERAL}: [-7.853982e-01, 0.000000e+00, 7.853982e-01]

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

// CHECK{LITERAL}: [7.853982e-01, 2.356194e+00, -2.356194e+00, -7.853982e-01]

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

// CHECK{LITERAL}: [8.660254e-01, 7.071068e-01]

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

// CHECK{LITERAL}: [1.543081e+00, 1.000000e+00, 1.543081e+00]

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

// CHECK{LITERAL}: [3.678794e-01, 1.000000e+00, 2.718282e+00]

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

// CHECK{LITERAL}: [-2.000000e+00, 0.000000e+00, 2.000000e+00]

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

// CHECK{LITERAL}: [-2.000000e+00, 0.000000e+00, 2.000000e+00]

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

// CHECK{LITERAL}: [-1.000000e+00, 0.000000e+00, 1.000000e+00]

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

// CHECK{LITERAL}: [5.000000e-01, 7.071068e-01]

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

// CHECK{LITERAL}: [-1.175201e+00, 0.000000e+00, 1.175201e+00]

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

// CHECK{LITERAL}: [2.000000e+00, 3.000000e+00]

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

// CHECK{LITERAL}: [5.773503e-01, 1.732051e+00]

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

// CHECK{LITERAL}: [-7.615942e-01, 0.000000e+00, 7.615942e-01]

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

func.func @main() -> () {
    call @test_abs() : () -> ()
    call @test_acos() : () -> ()
    call @test_asin() : () -> ()
    call @test_atan() : () -> ()
    call @test_atan2() : () -> ()
    call @test_cos() : () -> ()
    call @test_cosh() : () -> ()
    call @test_exp() : () -> ()
    call @test_log() : () -> ()
    call @test_log10() : () -> ()
    call @test_sign() : () -> ()
    call @test_sin() : () -> ()
    call @test_sinh() : () -> ()
    call @test_sqrt() : () -> ()
    call @test_tan() : () -> ()
    call @test_tanh() : () -> ()
    return
}
