// RUN: marco-opt %s --split-input-file --verify-diagnostics

// Array and scalar can't be added without the element-wise operator.

func.func @addArrayScalar(%arg0: tensor<3x!modelica.real>, %arg1: !modelica.real) {
    // expected-error @+2 {{operands must be both scalars or both arrays}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.add %arg0, %arg1 : (tensor<3x!modelica.real>, !modelica.real) -> tensor<3x!modelica.real>
    return
}

// -----

func.func @addMismatchingSizes(%arg0: tensor<3x!modelica.real>, %arg1: tensor<4x!modelica.real>) {
    // expected-error @+2 {{operands have incompatible shapes}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.add %arg0, %arg1 : (tensor<3x!modelica.real>, tensor<4x!modelica.real>) -> tensor<3x!modelica.real>
    return
}

// -----

func.func @addBoolean(%arg0: !modelica.bool) {
    // expected-error @+2 {{operands must be numeric}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.add %arg0, %arg0 : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return
}

// -----

// The result of a division is always Real.

func.func @divInteger(%arg0: !modelica.int) {
    // expected-error @+2 {{inferred type(s) '!modelica.real' are incompatible with return type(s) of operation '!modelica.int'}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.div %arg0, %arg0 : (!modelica.int, !modelica.int) -> !modelica.int
    return
}

// -----

func.func @divByArray(%arg0: tensor<3x!modelica.real>) {
    // expected-error @+2 {{expected a numeric dividend and a numeric scalar divisor}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.div %arg0, %arg0 : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> tensor<3x!modelica.real>
    return
}

// -----

func.func @mulIncompatibleMatrices(%arg0: tensor<2x3x!modelica.real>) {
    // expected-error @+2 {{incompatible dimensions for multiplication}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.mul %arg0, %arg0 : (tensor<2x3x!modelica.real>, tensor<2x3x!modelica.real>) -> tensor<2x3x!modelica.real>
    return
}

// -----

// Matrix exponentiation requires a square matrix and an Integer exponent.

func.func @powNonSquareMatrix(%arg0: tensor<2x3x!modelica.real>, %arg1: !modelica.int) {
    // expected-error @+2 {{exponentiation requires numeric scalars, or a square matrix and an Integer exponent}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.pow %arg0, %arg1 : (tensor<2x3x!modelica.real>, !modelica.int) -> tensor<2x3x!modelica.real>
    return
}

// -----

// Relational operators are defined only for scalars.

func.func @relationalArrays(%arg0: tensor<3x!modelica.real>) {
    // expected-error @+2 {{relational operators require scalar operands of comparable simple types}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.lt %arg0, %arg0 : (tensor<3x!modelica.real>, tensor<3x!modelica.real>) -> !modelica.bool
    return
}

// -----

func.func @relationalDifferentEnums(%arg0: !modelica.enum<@A>, %arg1: !modelica.enum<@B>) {
    // expected-error @+2 {{relational operators require scalar operands of comparable simple types}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.eq %arg0, %arg1 : (!modelica.enum<@A>, !modelica.enum<@B>) -> !modelica.bool
    return
}

// -----

func.func @ifExprIncompatibleBranch(%arg0: !modelica.bool, %arg1: !modelica.real) {
    // expected-error @+1 {{yielded type '!modelica.real' is not compatible with the result type '!modelica.int'}}
    %0 = modelica.if_expr %arg0 : !modelica.int {
        modelica.yield %arg1 : !modelica.real
    } else {
        modelica.yield %arg1 : !modelica.real
    }
    return
}

// -----

func.func @rangeBooleanWithStep(%arg0: !modelica.bool) {
    // expected-error @+2 {{a step can be specified only for Integer and Real ranges}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.range %arg0 to %arg0 step %arg0 : (!modelica.bool, !modelica.bool, !modelica.bool) -> tensor<?x!modelica.bool>
    return
}

// -----

func.func @arrayDifferentShapes(%arg0: tensor<2x!modelica.int>, %arg1: tensor<3x!modelica.int>) {
    // expected-error @+2 {{array elements must have the same shape}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.array %arg0, %arg1 : (tensor<2x!modelica.int>, tensor<3x!modelica.int>) -> tensor<2x?x!modelica.int>
    return
}

// -----

func.func @catOutOfBounds(%arg0: tensor<2x!modelica.int>) {
    %dim = modelica.constant #modelica<int 2> : !modelica.int
    // expected-error @+2 {{concatenation dimension 2 is out of bounds for arrays of rank 1}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.cat %dim, %arg0, %arg0 : (!modelica.int, tensor<2x!modelica.int>, tensor<2x!modelica.int>) -> tensor<4x!modelica.int>
    return
}

// -----

// The last iterator varies along the outermost dimension.

func.func @arrayForWrongShape(%arg0: tensor<2x!modelica.int>, %arg1: tensor<3x!modelica.int>) {
    // expected-error @+1 {{expected result type compatible with 'tensor<3x2x!modelica.int>', but got 'tensor<2x3x!modelica.int>'}}
    %0 = modelica.array_for (%i in %arg0 : tensor<2x!modelica.int>, %j in %arg1 : tensor<3x!modelica.int>) -> tensor<2x3x!modelica.int> {
        modelica.yield %i : !modelica.int
    }
    return
}

// -----

func.func @reductionProductOfArrays(%arg0: tensor<2x!modelica.int>) {
    // expected-error @+1 {{product requires scalar Integer or Real values}}
    %0 = modelica.reduction product (%i in %arg0 : tensor<2x!modelica.int>) -> tensor<2x!modelica.int> {
        modelica.yield %arg0 : tensor<2x!modelica.int>
    }
    return
}

// -----

func.func @arrayReduceSumBoolean(%arg0: tensor<2x!modelica.bool>) {
    // expected-error @+1 {{sum requires Integer or Real values}}
    %0 = modelica.array_reduce sum %arg0 : (tensor<2x!modelica.bool>) -> !modelica.bool
    return
}

// -----

func.func @minString(%arg0: !modelica.string) {
    // expected-error @+2 {{operands must be of comparable enumeration, Boolean, Integer or Real types}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.min %arg0, %arg0 : (!modelica.string, !modelica.string) -> !modelica.string
    return
}

// -----

// The derivative operator requires Real values.

func.func @derInteger(%arg0: !modelica.int) {
    // expected-error @+2 {{operand must be Real}}
    // expected-error @+1 {{failed to infer returned types}}
    %0 = modelica.der %arg0 : (!modelica.int) -> !modelica.int
    return
}

// -----

func.func @toStringFormatAndOptions(%arg0: !modelica.real, %arg1: !modelica.int, %arg2: !modelica.string) {
    // expected-error @+1 {{format can't be combined with the other options}}
    %0 = modelica.to_string %arg0 minimumLength %arg1 format %arg2 : (!modelica.real, !modelica.int, !modelica.string) -> !modelica.string
    return
}

// -----

func.func @toStringSignificantDigitsOfInteger(%arg0: !modelica.int) {
    // expected-error @+1 {{significantDigits is allowed only for Real values}}
    %0 = modelica.to_string %arg0 significantDigits %arg0 : (!modelica.int, !modelica.int) -> !modelica.string
    return
}

// -----

func.func @integerToEnumNonEnum(%arg0: !modelica.int) {
    // expected-error @+1 {{result must be an enumeration}}
    %0 = modelica.integer_to_enum %arg0 : (!modelica.int) -> !modelica.int
    return
}

// -----

func.func @promoteLowerRank(%arg0: tensor<2x3x!modelica.int>) {
    // expected-error @+1 {{rank must be greater than or equal to the rank of the operand}}
    %0 = modelica.promote %arg0, 1 : (tensor<2x3x!modelica.int>) -> tensor<2x!modelica.int>
    return
}

// -----

func.func @constantArrayWrongCount() {
    // expected-error @+1 {{expected 3 elements, but got 2}}
    %0 = modelica.constant #modelica<array [1, 2]> : tensor<3x!modelica.int>
    return
}

// -----

func.func @constantLiteralWrongType() {
    // expected-error @+1 {{invalid type '!modelica.real' for the literal}}
    %0 = modelica.constant #modelica<int 1> : !modelica.real
    return
}

// -----

func.func @enumLiteralZeroOrdinal() {
    // expected-error @+1 {{the ordinal value of an enumeration literal must be greater than zero}}
    %0 = modelica.constant #modelica<enum "a" = 0> : !modelica.enum<@E>
    return
}
