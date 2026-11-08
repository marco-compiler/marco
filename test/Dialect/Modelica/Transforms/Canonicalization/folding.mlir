// RUN: marco-opt %s --split-input-file --canonicalize | FileCheck %s

// Integer and Real arithmetic, with implicit conversion of Integer values to
// Real.

// CHECK-LABEL: @arithmetic
// CHECK-DAG: %[[add:.*]] = modelica.constant #modelica<int 10> : !modelica.int
// CHECK-DAG: %[[addReal:.*]] = modelica.constant #modelica<real 9.500000e+00> : !modelica.real
// CHECK-DAG: %[[sub:.*]] = modelica.constant #modelica<int -4> : !modelica.int
// CHECK-DAG: %[[mul:.*]] = modelica.constant #modelica<int 21> : !modelica.int
// CHECK-DAG: %[[div:.*]] = modelica.constant #modelica<real 3.500000e+00> : !modelica.real
// CHECK: return %[[add]], %[[addReal]], %[[sub]], %[[mul]], %[[div]]

func.func @arithmetic() -> (!modelica.int, !modelica.real, !modelica.int, !modelica.int, !modelica.real) {
    %x = modelica.constant #modelica<int 7> : !modelica.int
    %y = modelica.constant #modelica<int 3> : !modelica.int
    %z = modelica.constant #modelica<real 2.0> : !modelica.real
    %w = modelica.constant #modelica<real 2.5> : !modelica.real
    %0 = modelica.add %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.add %x, %w : (!modelica.int, !modelica.real) -> !modelica.real
    %2 = modelica.sub %y, %x : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.mul %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    %4 = modelica.div %x, %z : (!modelica.int, !modelica.real) -> !modelica.real
    return %0, %1, %2, %3, %4 : !modelica.int, !modelica.real, !modelica.int, !modelica.int, !modelica.real
}

// -----

// Operations whose result is undefined or not representable are not folded.

// CHECK-LABEL: @notFolded
// CHECK: modelica.div
// CHECK: modelica.add
// CHECK: modelica.div_trunc

func.func @notFolded() -> (!modelica.real, !modelica.int, !modelica.int) {
    %zero = modelica.constant #modelica<int 0> : !modelica.int
    %one = modelica.constant #modelica<int 1> : !modelica.int
    %max = modelica.constant #modelica<int 9223372036854775807> : !modelica.int
    %0 = modelica.div %one, %zero : (!modelica.int, !modelica.int) -> !modelica.real
    %1 = modelica.add %max, %one : (!modelica.int, !modelica.int) -> !modelica.int
    %2 = modelica.div_trunc %one, %zero : (!modelica.int, !modelica.int) -> !modelica.int
    return %0, %1, %2 : !modelica.real, !modelica.int, !modelica.int
}

// -----

// Unitful literals keep their unit when negated, so that
// the implicit unit conversion of binding equations can still be applied.
// Other operations are not folded, as the unit would be lost.

// CHECK-LABEL: @units
// CHECK-DAG: %[[neg:.*]] = modelica.constant #modelica<int -5 unit "cm"> : !modelica.int
// CHECK-DAG: %[[add:.*]] = modelica.add
// CHECK: return %[[neg]], %[[add]]

func.func @units() -> (!modelica.int, !modelica.int) {
    %x = modelica.constant #modelica<int 5 unit "cm"> : !modelica.int
    %0 = modelica.neg %x : (!modelica.int) -> !modelica.int
    %1 = modelica.add %x, %x : (!modelica.int, !modelica.int) -> !modelica.int
    return %0, %1 : !modelica.int, !modelica.int
}

// -----

// Exponentiation.

// CHECK-LABEL: @pow
// CHECK-DAG: %[[c0:.*]] = modelica.constant #modelica<real 1.000000e+00> : !modelica.real
// CHECK-DAG: %[[c1:.*]] = modelica.constant #modelica<real -8.000000e+00> : !modelica.real
// CHECK-DAG: %[[c2:.*]] = modelica.constant #modelica<real 5.000000e-01> : !modelica.real
// CHECK-DAG: %[[c3:.*]] = modelica.pow %{{.*}}, %{{.*}} : (!modelica.int, !modelica.real)
// CHECK: return %[[c0]], %[[c1]], %[[c2]], %[[c3]]

func.func @pow() -> (!modelica.real, !modelica.real, !modelica.real, !modelica.real) {
    %zero = modelica.constant #modelica<int 0> : !modelica.int
    %minusTwo = modelica.constant #modelica<int -2> : !modelica.int
    %three = modelica.constant #modelica<int 3> : !modelica.int
    %minusOne = modelica.constant #modelica<int -1> : !modelica.int
    %two = modelica.constant #modelica<int 2> : !modelica.int
    %half = modelica.constant #modelica<real 0.5> : !modelica.real
    // x ^ 0 = 1 for any x, including 0.
    %0 = modelica.pow %zero, %zero : (!modelica.int, !modelica.int) -> !modelica.real
    // Negative base with Integer exponent.
    %1 = modelica.pow %minusTwo, %three : (!modelica.int, !modelica.int) -> !modelica.real
    // Negative Integer exponent gives a non-integer result.
    %2 = modelica.pow %two, %minusOne : (!modelica.int, !modelica.int) -> !modelica.real
    // Negative base with non-integer Real exponent is illegal.
    %3 = modelica.pow %minusTwo, %half : (!modelica.int, !modelica.real) -> !modelica.real
    return %0, %1, %2, %3 : !modelica.real, !modelica.real, !modelica.real, !modelica.real
}

// -----

// Integer division, modulus and remainder.

// CHECK-LABEL: @integerDivision
// CHECK-DAG: %[[div:.*]] = modelica.constant #modelica<int -1> : !modelica.int
// CHECK-DAG: %[[mod:.*]] = modelica.constant #modelica<int 1> : !modelica.int
// CHECK-DAG: %[[rem:.*]] = modelica.constant #modelica<int -3> : !modelica.int
// CHECK-DAG: %[[modReal:.*]] = modelica.constant #modelica<real 1.500000e+00> : !modelica.real
// CHECK: return %[[div]], %[[mod]], %[[rem]], %[[modReal]]

func.func @integerDivision() -> (!modelica.int, !modelica.int, !modelica.int, !modelica.real) {
    %x = modelica.constant #modelica<int -7> : !modelica.int
    %y = modelica.constant #modelica<int 4> : !modelica.int
    %z = modelica.constant #modelica<real -2.5> : !modelica.real
    %0 = modelica.div_trunc %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    %1 = modelica.mod %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    %2 = modelica.rem %x, %y : (!modelica.int, !modelica.int) -> !modelica.int
    %3 = modelica.mod %z, %y : (!modelica.real, !modelica.int) -> !modelica.real
    return %0, %1, %2, %3 : !modelica.int, !modelica.int, !modelica.int, !modelica.real
}

// -----

// Element-wise folding with broadcasting of scalars.

// CHECK-LABEL: @elementwise
// CHECK-DAG: %[[c0:.*]] = modelica.constant #modelica<array [1.100000e+01, 1.200000e+01, 1.300000e+01]> : tensor<3x!modelica.real>
// CHECK-DAG: %[[c1:.*]] = modelica.constant #modelica<array [-1, -2, -3]> : tensor<3x!modelica.int>
// CHECK-DAG: %[[c2:.*]] = modelica.constant #modelica<array [1, 4, 9]> : tensor<3x!modelica.int>
// CHECK: return %[[c0]], %[[c1]], %[[c2]]

func.func @elementwise() -> (tensor<3x!modelica.real>, tensor<3x!modelica.int>, tensor<3x!modelica.int>) {
    %a = modelica.constant #modelica<array [1, 2, 3]> : tensor<3x!modelica.int>
    %s = modelica.constant #modelica<real 10.0> : !modelica.real
    %0 = modelica.add_ew %s, %a : (!modelica.real, tensor<3x!modelica.int>) -> tensor<3x!modelica.real>
    %1 = modelica.neg %a : (tensor<3x!modelica.int>) -> tensor<3x!modelica.int>
    %2 = modelica.mul_ew %a, %a : (tensor<3x!modelica.int>, tensor<3x!modelica.int>) -> tensor<3x!modelica.int>
    return %0, %1, %2 : tensor<3x!modelica.real>, tensor<3x!modelica.int>, tensor<3x!modelica.int>
}

// -----

// String concatenation.

// CHECK-LABEL: @concatenation
// CHECK: %[[c:.*]] = modelica.constant #modelica<string "ab"> : !modelica.string
// CHECK: return %[[c]]

func.func @concatenation() -> !modelica.string {
    %a = modelica.constant #modelica<string "a"> : !modelica.string
    %b = modelica.constant #modelica<string "b"> : !modelica.string
    %0 = modelica.add %a, %b : (!modelica.string, !modelica.string) -> !modelica.string
    return %0 : !modelica.string
}

// -----

// Relational operators on all the simple types.

// CHECK-LABEL: @relational
// CHECK-DAG: %[[true:.*]] = modelica.constant #modelica<bool true> : !modelica.bool
// CHECK-DAG: %[[false:.*]] = modelica.constant #modelica<bool false> : !modelica.bool
// CHECK: return %[[true]], %[[true]], %[[true]], %[[true]], %[[false]]

func.func @relational() -> (!modelica.bool, !modelica.bool, !modelica.bool, !modelica.bool, !modelica.bool) {
    %i = modelica.constant #modelica<int 2> : !modelica.int
    %r = modelica.constant #modelica<real 2.5> : !modelica.real
    %f = modelica.constant #modelica<bool false> : !modelica.bool
    %t = modelica.constant #modelica<bool true> : !modelica.bool
    %s1 = modelica.constant #modelica<string "abc"> : !modelica.string
    %s2 = modelica.constant #modelica<string "abd"> : !modelica.string
    %e1 = modelica.constant #modelica<enum "a" = 1> : !modelica.enum<@E>
    %e2 = modelica.constant #modelica<enum "b" = 2> : !modelica.enum<@E>
    %0 = modelica.lt %i, %r : (!modelica.int, !modelica.real) -> !modelica.bool
    %1 = modelica.lt %f, %t : (!modelica.bool, !modelica.bool) -> !modelica.bool
    %2 = modelica.le %s1, %s2 : (!modelica.string, !modelica.string) -> !modelica.bool
    %3 = modelica.gt %e2, %e1 : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.bool
    %4 = modelica.eq %e1, %e2 : (!modelica.enum<@E>, !modelica.enum<@E>) -> !modelica.bool
    return %0, %1, %2, %3, %4 : !modelica.bool, !modelica.bool, !modelica.bool, !modelica.bool, !modelica.bool
}

// -----

// CHECK-LABEL: @logical
// CHECK-DAG: %[[c0:.*]] = modelica.constant #modelica<array [false, true]> : tensor<2x!modelica.bool>
// CHECK-DAG: %[[c1:.*]] = modelica.constant #modelica<bool false> : !modelica.bool
// CHECK-DAG: %[[c2:.*]] = modelica.constant #modelica<bool true> : !modelica.bool
// CHECK: return %[[c0]], %[[c1]], %[[c2]]

func.func @logical() -> (tensor<2x!modelica.bool>, !modelica.bool, !modelica.bool) {
    %a = modelica.constant #modelica<array [true, false]> : tensor<2x!modelica.bool>
    %t = modelica.constant #modelica<bool true> : !modelica.bool
    %f = modelica.constant #modelica<bool false> : !modelica.bool
    %0 = modelica.not %a : (tensor<2x!modelica.bool>) -> tensor<2x!modelica.bool>
    %1 = modelica.and %t, %f : (!modelica.bool, !modelica.bool) -> !modelica.bool
    %2 = modelica.or %t, %f : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return %0, %1, %2 : tensor<2x!modelica.bool>, !modelica.bool, !modelica.bool
}

// -----

// If-expressions with constant conditions are replaced by the selected
// branch, unless the implicit conversion to Real would be needed.

// CHECK-LABEL: @ifExpr
// CHECK-SAME: (%[[x:.*]]: !modelica.real, %[[y:.*]]: !modelica.int)
// CHECK: %[[converted:.*]] = modelica.if_expr
// CHECK: return %[[x]], %[[converted]]

func.func @ifExpr(%arg0: !modelica.real, %arg1: !modelica.int) -> (!modelica.real, !modelica.real) {
    %t = modelica.constant #modelica<bool true> : !modelica.bool
    %f = modelica.constant #modelica<bool false> : !modelica.bool
    %0 = modelica.if_expr %t : !modelica.real {
        modelica.yield %arg0 : !modelica.real
    } else {
        modelica.yield %arg1 : !modelica.int
    }
    %1 = modelica.if_expr %f : !modelica.real {
        modelica.yield %arg0 : !modelica.real
    } else {
        modelica.yield %arg1 : !modelica.int
    }
    return %0, %1 : !modelica.real, !modelica.real
}

// -----

// Ranges. Enumeration ranges and ranges with a
// dynamically sized result type are not folded.

// CHECK-LABEL: @range
// CHECK-DAG: %[[int:.*]] = modelica.constant #modelica<array [1, 3, 5]> : tensor<3x!modelica.int>
// CHECK-DAG: %[[real:.*]] = modelica.constant #modelica<array [2.700000e+00, 3.700000e+00, 4.700000e+00, 5.700000e+00, 6.700000e+00]> : tensor<5x!modelica.real>
// CHECK-DAG: %[[bool:.*]] = modelica.constant #modelica<array [false, true]> : tensor<2x!modelica.bool>
// CHECK-DAG: %[[empty:.*]] = modelica.constant #modelica<array []> : tensor<0x!modelica.int>
// CHECK-DAG: %[[enum:.*]] = modelica.range {{.*}} -> tensor<2x!modelica.enum<@E>>
// CHECK-DAG: %[[dynamic:.*]] = modelica.range {{.*}} -> tensor<?x!modelica.int>
// CHECK: return %[[int]], %[[real]], %[[bool]], %[[empty]], %[[enum]], %[[dynamic]]

func.func @range() -> (tensor<3x!modelica.int>, tensor<5x!modelica.real>, tensor<2x!modelica.bool>, tensor<0x!modelica.int>, tensor<2x!modelica.enum<@E>>, tensor<?x!modelica.int>) {
    %one = modelica.constant #modelica<int 1> : !modelica.int
    %two = modelica.constant #modelica<int 2> : !modelica.int
    %six = modelica.constant #modelica<int 6> : !modelica.int
    %a = modelica.constant #modelica<real 2.7> : !modelica.real
    %b = modelica.constant #modelica<real 6.8> : !modelica.real
    %f = modelica.constant #modelica<bool false> : !modelica.bool
    %t = modelica.constant #modelica<bool true> : !modelica.bool
    %e1 = modelica.constant #modelica<enum "a" = 1> : !modelica.enum<@E>
    %e2 = modelica.constant #modelica<enum "b" = 2> : !modelica.enum<@E>
    // 1:2:6 = {1, 3, 5}
    %0 = modelica.range %one to %six step %two : (!modelica.int, !modelica.int, !modelica.int) -> tensor<3x!modelica.int>
    // 2.7:6.8 = {2.7, 3.7, 4.7, 5.7, 6.7}
    %1 = modelica.range %a to %b : (!modelica.real, !modelica.real) -> tensor<5x!modelica.real>
    %2 = modelica.range %f to %t : (!modelica.bool, !modelica.bool) -> tensor<2x!modelica.bool>
    %3 = modelica.range %six to %one : (!modelica.int, !modelica.int) -> tensor<0x!modelica.int>
    %4 = modelica.range %e1 to %e2 : (!modelica.enum<@E>, !modelica.enum<@E>) -> tensor<2x!modelica.enum<@E>>
    %5 = modelica.range %one to %six : (!modelica.int, !modelica.int) -> tensor<?x!modelica.int>
    return %0, %1, %2, %3, %4, %5 : tensor<3x!modelica.int>, tensor<5x!modelica.real>, tensor<2x!modelica.bool>, tensor<0x!modelica.int>, tensor<2x!modelica.enum<@E>>, tensor<?x!modelica.int>
}

// -----

// Array constructor with Integer and Real elements, and nested arrays.

// CHECK-LABEL: @array
// CHECK-DAG: %[[c0:.*]] = modelica.constant #modelica<array [1.000000e+00, 2.500000e+00]> : tensor<2x!modelica.real>
// CHECK-DAG: %[[c1:.*]] = modelica.constant #modelica<array [1, 2, 1, 2]> : tensor<2x2x!modelica.int>
// CHECK: return %[[c0]], %[[c1]]

func.func @array() -> (tensor<2x!modelica.real>, tensor<2x2x!modelica.int>) {
    %i = modelica.constant #modelica<int 1> : !modelica.int
    %r = modelica.constant #modelica<real 2.5> : !modelica.real
    %v = modelica.constant #modelica<array [1, 2]> : tensor<2x!modelica.int>
    %0 = modelica.array %i, %r : (!modelica.int, !modelica.real) -> tensor<2x!modelica.real>
    %1 = modelica.array %v, %v : (tensor<2x!modelica.int>, tensor<2x!modelica.int>) -> tensor<2x2x!modelica.int>
    return %0, %1 : tensor<2x!modelica.real>, tensor<2x2x!modelica.int>
}

// -----

// Dimension functions fold based on the static shape, even if the array is
// not a constant.

// CHECK-LABEL: @dimensions
// CHECK-DAG: %[[ndims:.*]] = modelica.constant #modelica<int 2> : !modelica.int
// CHECK-DAG: %[[size:.*]] = modelica.constant #modelica<array [2, 3]> : tensor<2x!modelica.int>
// CHECK-DAG: %[[size2:.*]] = modelica.constant #modelica<int 3> : !modelica.int
// CHECK-DAG: %[[dynamic:.*]] = modelica.size %{{.*}}, %{{.*}} : (tensor<?x?x!modelica.real>, !modelica.int) -> !modelica.int
// CHECK: return %[[ndims]], %[[size]], %[[size2]], %[[dynamic]]

func.func @dimensions(%arg0: tensor<2x3x!modelica.real>, %arg1: tensor<?x?x!modelica.real>) -> (!modelica.int, tensor<2x!modelica.int>, !modelica.int, !modelica.int) {
    %one = modelica.constant #modelica<int 1> : !modelica.int
    %0 = modelica.ndims %arg0 : (tensor<2x3x!modelica.real>) -> !modelica.int
    %1 = modelica.size %arg0 : (tensor<2x3x!modelica.real>) -> tensor<2x!modelica.int>
    %2 = modelica.size %arg0, %one : (tensor<2x3x!modelica.real>, !modelica.int) -> !modelica.int
    %3 = modelica.size %arg1, %one : (tensor<?x?x!modelica.real>, !modelica.int) -> !modelica.int
    return %0, %1, %2, %3 : !modelica.int, tensor<2x!modelica.int>, !modelica.int, !modelica.int
}

// -----

// Array reductions.

// CHECK-LABEL: @arrayReduce
// CHECK-DAG: %[[sum:.*]] = modelica.constant #modelica<int 10> : !modelica.int
// CHECK-DAG: %[[product:.*]] = modelica.constant #modelica<int 24> : !modelica.int
// CHECK-DAG: %[[min:.*]] = modelica.constant #modelica<int 1> : !modelica.int
// CHECK-DAG: %[[max:.*]] = modelica.constant #modelica<int 4> : !modelica.int
// CHECK: return %[[sum]], %[[product]], %[[min]], %[[max]]

func.func @arrayReduce() -> (!modelica.int, !modelica.int, !modelica.int, !modelica.int) {
    %a = modelica.constant #modelica<array [3, 1, 4, 2]> : tensor<2x2x!modelica.int>
    %0 = modelica.array_reduce sum %a : (tensor<2x2x!modelica.int>) -> !modelica.int
    %1 = modelica.array_reduce product %a : (tensor<2x2x!modelica.int>) -> !modelica.int
    %2 = modelica.array_reduce min %a : (tensor<2x2x!modelica.int>) -> !modelica.int
    %3 = modelica.array_reduce max %a : (tensor<2x2x!modelica.int>) -> !modelica.int
    return %0, %1, %2, %3 : !modelica.int, !modelica.int, !modelica.int, !modelica.int
}

// -----

// Numeric functions.

// CHECK-LABEL: @numeric
// CHECK-DAG: %[[abs:.*]] = modelica.constant #modelica<int 7> : !modelica.int
// CHECK-DAG: %[[sign:.*]] = modelica.constant #modelica<int -1> : !modelica.int
// CHECK-DAG: %[[floor:.*]] = modelica.constant #modelica<real -3.000000e+00> : !modelica.real
// CHECK-DAG: %[[integer:.*]] = modelica.constant #modelica<int -3> : !modelica.int
// CHECK-DAG: %[[root:.*]] = modelica.constant #modelica<real -2.000000e+00> : !modelica.real
// CHECK-DAG: %[[min:.*]] = modelica.constant #modelica<real -7.000000e+00> : !modelica.real
// CHECK-DAG: %[[ordinal:.*]] = modelica.constant #modelica<int 2> : !modelica.int
// CHECK-DAG: %[[sqrt:.*]] = modelica.sqrt
// CHECK: return %[[abs]], %[[sign]], %[[floor]], %[[integer]], %[[root]], %[[min]], %[[ordinal]], %[[sqrt]]

func.func @numeric() -> (!modelica.int, !modelica.int, !modelica.real, !modelica.int, !modelica.real, !modelica.real, !modelica.int, !modelica.real) {
    %x = modelica.constant #modelica<int -7> : !modelica.int
    %r = modelica.constant #modelica<real -2.5> : !modelica.real
    %m = modelica.constant #modelica<real -8.0> : !modelica.real
    %three = modelica.constant #modelica<int 3> : !modelica.int
    %e = modelica.constant #modelica<enum "b" = 2> : !modelica.enum<@E>
    %0 = modelica.abs %x : (!modelica.int) -> !modelica.int
    %1 = modelica.sign %r : (!modelica.real) -> !modelica.int
    %2 = modelica.floor %r : (!modelica.real) -> !modelica.real
    %3 = modelica.integer %r : (!modelica.real) -> !modelica.int
    // Odd roots of negative values are defined.
    %4 = modelica.nth_root %m, %three : (!modelica.real, !modelica.int) -> !modelica.real
    %5 = modelica.min %x, %r : (!modelica.int, !modelica.real) -> !modelica.real
    %6 = modelica.enum_to_integer %e : (!modelica.enum<@E>) -> !modelica.int
    // The square root of a negative value is not folded.
    %7 = modelica.sqrt %r : (!modelica.real) -> !modelica.real
    return %0, %1, %2, %3, %4, %5, %6, %7 : !modelica.int, !modelica.int, !modelica.real, !modelica.int, !modelica.real, !modelica.real, !modelica.int, !modelica.real
}

// -----

// CHECK-LABEL: @elementary
// CHECK-DAG: %[[zero:.*]] = modelica.constant #modelica<real 0.000000e+00> : !modelica.real
// CHECK-DAG: %[[one:.*]] = modelica.constant #modelica<real 1.000000e+00> : !modelica.real
// CHECK-DAG: %[[log:.*]] = modelica.log
// CHECK: return %[[zero]], %[[one]], %[[log]]

func.func @elementary() -> (!modelica.real, !modelica.real, !modelica.real) {
    %zero = modelica.constant #modelica<int 0> : !modelica.int
    %0 = modelica.sin %zero : (!modelica.int) -> !modelica.real
    %1 = modelica.exp %zero : (!modelica.int) -> !modelica.real
    %2 = modelica.log %zero : (!modelica.int) -> !modelica.real
    return %0, %1, %2 : !modelica.real, !modelica.real, !modelica.real
}

// -----

// CHECK-LABEL: @eventRelated
// CHECK: %[[c:.*]] = modelica.constant #modelica<real 1.500000e+00> : !modelica.real
// CHECK: return %[[c]], %[[c]]

func.func @eventRelated() -> (!modelica.real, !modelica.real) {
    %x = modelica.constant #modelica<real 1.5> : !modelica.real
    %p = modelica.constant #modelica<int 1> : !modelica.int
    %0 = modelica.no_event %x : (!modelica.real) -> !modelica.real
    %1 = modelica.smooth %p, %x : (!modelica.int, !modelica.real) -> !modelica.real
    return %0, %1 : !modelica.real, !modelica.real
}
