// RUN: marco-opt %s --split-input-file --verify-diagnostics

func.func @memberOfScalar(%x: !modelica.real) {
    // expected-error @+1 {{operand must be an object or an array of objects}}
    %0 = modelica.member %x, @y : (!modelica.real) -> !modelica.real
    return
}

// -----

func.func @memberSliceShape(%x: tensor<3x!modelica.object<@R>>) {
    // expected-error @+1 {{result must have at least the dimensions of the array of objects}}
    %0 = modelica.member %x, @y : (tensor<3x!modelica.object<@R>>) -> !modelica.real
    return
}

// -----

func.func @subscriptTooMany(%a: tensor<3x!modelica.real>, %i: !modelica.int) {
    // expected-error @+1 {{too many subscripts for an array of rank 1}}
    %0 = modelica.subscript %a[%i, %i] : (tensor<3x!modelica.real>, !modelica.int, !modelica.int) -> !modelica.real
    return
}

// -----

func.func @subscriptRealIndex(%a: tensor<3x!modelica.real>, %i: !modelica.real) {
    // expected-error @+1 {{indices must be Integer, Boolean or enumeration values}}
    %0 = modelica.subscript %a[%i] : (tensor<3x!modelica.real>, !modelica.real) -> !modelica.real
    return
}

// -----

func.func @subscriptWrongResult(%a: tensor<3x4x!modelica.real>, %i: !modelica.int) {
    // expected-error @+1 {{expected result type compatible with 'tensor<4x!modelica.real>', but got '!modelica.real'}}
    %0 = modelica.subscript %a[%i] : (tensor<3x4x!modelica.real>, !modelica.int) -> !modelica.real
    return
}

// -----

func.func @callDuplicateNames(%x: !modelica.real) {
    // expected-error @+1 {{duplicate named argument "y"}}
    %0 = modelica.call @f(@y = %x, @y = %x) : (!modelica.real, !modelica.real) -> !modelica.real
    return
}

// -----

func.func @callPositionalAfterNamed(%x: !modelica.real) {
    // expected-error @+1 {{positional arguments must precede the named ones}}
    %0 = modelica.call @f(@y = %x, %x) : (!modelica.real, !modelica.real) -> !modelica.real
    return
}

// -----

func.func @inStreamOfValue(%x: !modelica.real) {
    // expected-error @+1 {{stream variable must be a component reference}}
    %0 = modelica.in_stream %x : (!modelica.real) -> !modelica.real
    return
}
