#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "marco/Dialect/Modelica/ModelicaAttributes.td"
include "marco/Dialect/Modelica/ModelicaDialect.td"
include "marco/Dialect/Modelica/ModelicaInterfaces.td"
include "marco/Dialect/Modelica/ModelicaTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// Modelica Boolean type
def ModelicaBoolean : Type<CPred<"$_self.isa<::mlir::modelica::BooleanType>()">,
                           "boolean", "::mlir::modelica::BooleanType">;

// Modelica Boolean attribute
def ModelicaBooleanAttr : Attr<CPred<"$_self.isa<::mlir::modelica::BoolAttr>()">, "boolean attribute"> {
  let storageType = [{ ::mlir::modelica::BoolAttr }];
  let returnType = [{ bool }];
  let valueType = ModelicaBoolean;
  let constBuilderCall = "::mlir::modelica::BooleanAttr::get($_builder.getContext(), $0)";
}

// Modelica Integer type
def ModelicaInteger : Type<CPred<"$_self.isa<::mlir::modelica::IntegerType>()">,
                                 "integer", "::mlir::modelica::IntegerType">;

// Modelica Integer attribute
def ModelicaIntegerAttr : Attr<CPred<"$_self.isa<::mlir::modelica::IntegerAttr>()">, "integer attribute"> {
  let storageType = [{ ::mlir::modelica::IntegerAttr }];
  let returnType = [{ long }];
  let valueType = ModelicaInteger;
  let constBuilderCall = "::mlir::modelica::IntegerAttr::get($_builder.getContext(), $0)";
}

// Modelica Real type
def ModelicaReal : Type<CPred<"$_self.isa<::mlir::modelica::RealType>()">,
                        "real", "::mlir::modelica::RealType">;

// Modelica Real attribute
def ModelicaRealAttr : Attr<CPred<"$_self.isa<::mlir::modelica::RealAttr>()">, "real attribute"> {
  let storageType = [{ ::mlir::modelica::RealAttr }];
  let returnType = [{ double }];
  let valueType = ModelicaReal;
  let constBuilderCall = "::mlir::modelica::RealAttr::get($_builder.getContext(), $0)";
}

// Modelica Array type
def ModelicaArray : Type<CPred<"$_self.isa<::mlir::modelica::ArrayType>()">,
                         "array", "::mlir::modelica::ArrayType">;

// Modelica-compatible scalar type
def AnyModelicaScalar : AnyTypeOf<[ModelicaBoolean, ModelicaInteger, ModelicaReal, Index]>;

// Whether a type is an ArrayType.
def IsArrayTypePred : CPred<"$_self.isa<::mlir::modelica::ArrayType>()">;

class ArrayOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsArrayTypePred,
                  "$_self.cast<::mlir::modelica::ArrayType>().getElementType()",
                  "array", "::mlir::modelica::ArrayType">;

def AnyArray : ArrayOf<[ModelicaBoolean, ModelicaInteger, ModelicaReal, Index, ModelicaArray]>;

// Modelica Tensor Array type
def ModelicaTensorArray : Type<CPred<"$_self.isa<::mlir::modelica::TensorArrayType>()">,
                                     "tensor_array", "::mlir::modelica::TensorArrayType">;

// Whether a type is an TensorArrayType.
def IsTensorArrayTypePred : CPred<"$_self.isa<::mlir::modelica::TensorArrayType>()">;

class TensorArrayOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsTensorArrayTypePred,
                  "$_self.cast<::mlir::modelica::TensorArrayType>().getElementType()",
                  "tensor_array", "::mlir::modelica::TensorArrayType">;

def AnyTensorArray : TensorArrayOf<[ModelicaBoolean, ModelicaInteger, ModelicaReal, Index]>;

// Whether a type is a MemberType.
def IsMemberTypePred : CPred<"$_self.isa<::mlir::modelica::MemberType>()">;

class MemberOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsMemberTypePred,
                  "$_self.cast<::mlir::modelica::MemberType>().getElementType()",
                  "member", "::mlir::modelica::MemberType">;

def AnyMember : MemberOf<[ModelicaBoolean, ModelicaInteger, ModelicaReal, Index]>;

//===----------------------------------------------------------------------===//
// Modelica operation definitions
//===----------------------------------------------------------------------===//

class Modelica_Op<string mnemonic, list<Trait> traits = []> :
    Op<Modelica_Dialect, mnemonic, traits>;

// Base class for unary operations.
class Modelica_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

// Base class for binary operations.
class Modelica_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

def Modelica_AllocaOp : Modelica_Op<"alloca",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Allocate an array on the stack.";

    let description = [{
        Allocate an array on the stack.
        The operation takes a variadic number of arguments, each of one representing
        the size of a dynamic dimension within the desired array type. Note that
        the amount of operands must match the number of dynamic dimensions and they
        all must be of `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` qualified(type($array))
    }];

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];

    let hasVerifier = 1;
}

def Modelica_AllocOp : Modelica_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Allocate an array on the heap.";

    let description = [{
        Allocate an array on the heap.
        The operation takes a variadic number of arguments, each of one representing
        the size of a dynamic dimension within the desired array type. Note that
        the amount of operands must match the number of dynamic dimensions and they
        all must be of `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` qualified(type($array))
    }];

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];

    let hasVerifier = 1;
}

def Modelica_ArrayCastOp : Modelica_Op<"array_cast", [ViewLikeOpInterface]>
{
    let summary = "Cast operation for array types.";

    let description = [{
        This operation should be used for only one purpose, that is to generalize the
        dimensions of an array. This can be done by either converting some dimensions
        to unknown ones or by converting the whole array to an unsized one.
        The operation is NOT intended to cast the element type to a different one or
        to to specialize the shape to a an array with dynamic dimensions.

        Correct usage examples:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<?x?x!modelica.int>
        %3 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<*x!modelica.int>
        ```

        Wrong usage examples:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$source);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $source attr-dict `:` qualified(type($source)) `->` qualified(type($result))
    }];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // ViewLikeOpInterface methods
        //===------------------------------------------------------------------===//

        Value getViewSource() {
            return getSource();
        }
    }];
}

def Modelica_ArrayFillOp : Modelica_Op<"array_fill",
    [TypesMatchWith<"type of 'value' matches element type of 'array'",
                    "array", "value",
                    "$_self.cast<ArrayType>().getElementType()">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Fill an array with a value";

    let description = [{
        Set all the elements of an array to a given value.
        The value to be set must have the same type of array elements.

        Example:

        ```mlir
        %value = ... : !modelica.real
        %array = ... : !modelica.array<?x!modelica.real>
        modelica.array_fill %array, %value : !modelica.array<?x!modelica.real>, !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$array, AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $array `,` $value attr-dict `:` qualified(type($array))
    }];
}

def Modelica_DimOp : Modelica_Op<"dim", [NoMemoryEffect]>
{
    let summary = "Get the size of an array dimension.";

    let description = [{
        Get the size of an array dimension. The operation is trivial in case of
        statically shaped arrays, while it is more useful in case of dimensions
        that are unknown at compile-time.
        The result value has `index` type.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x?x?x!modelica.int>
        %c2 = constant 2 : index
        %dim = modelica.dim %array, $c2 : !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array, Index:$dimension);
    let results = (outs Index:$result);

    let assemblyFormat = [{
        $array `,` $dimension attr-dict `:` qualified(type($array))
    }];

    let builders = [
        OpBuilder<(ins "Value":$array, "Value":$dimension), [{
            auto indexType = $_builder.getIndexType();
            build($_builder, $_state, indexType, array, dimension);
        }]>
    ];
}

def Modelica_FreeOp : Modelica_Op<"free", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocate an heap-allocated array.";

    let description = [{
        Deallocate an heap-allocated array.
        The array must have been created using the `modelica.alloc` operation.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x!modelica.int>
        modelica.free %array : !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs);

    let assemblyFormat = [{
        $array attr-dict `:` qualified(type($array))
    }];
}

def Modelica_LoadOp : Modelica_Op<"load",
    [TypesMatchWith<"result type matches element type of 'array'",
                     "array", "result",
                     "$_self.cast<ArrayType>().getElementType()">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Load a value from an array.";

    let description = [{
        Load a value from an array location given by indices. The number of
        arguments provided within brackets must match the rank of the array.

        Example:

        ```mlir
        $array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = modelica.load $array[%0, %1] : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array, Variadic<Index>:$indices);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $indices `]` attr-dict `:` qualified(type($array))
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "Value":$array, CArg<"ValueRange", "{}">:$indices), [{
            auto arrayType = array.getType().cast<ArrayType>();
            $_state.addOperands(array);
            $_state.addOperands(indices);
            $_state.types.push_back(arrayType.getElementType());
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_StoreOp : Modelica_Op<"store",
    [TypesMatchWith<"type of 'value' matches element type of 'array'",
                    "array", "value",
                    "$_self.cast<ArrayType>().getElementType()">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Store a value into an array.";

    let description = [{
        Store a value to an array location given by indices. The value stored must
        have the same type as the elemental type of the array. The number of
        arguments provided within brackets must match the rank of the array.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = ... : !modelica.int
        modelica.store %array[%0, %1], %value : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyType:$value,
        AnyArray:$array,
        Variadic<Index>:$indices);

    let results = (outs);

    let assemblyFormat = [{
        $array `[` $indices `]` `,` $value attr-dict `:` qualified(type($array))
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_SubscriptionOp : Modelica_Op<"subscription",
    [ViewLikeOpInterface,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Create a reduced-rank view of an array.";

    let description = [{
        Create a reduced-rank view of an array.

        Example:

        ```mlir
        %array = ... : !modelica.array<3x5x4x!modelica.int>
        %c1 : constant 1 : index
        %c2 : constant 2 : index
        %view = modelica.subscription $array[%c1, %c2] : !modelica.array<3x5x4x!modelica.int>
        // %view has type !modelica.array<4x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$source, Variadic<AnyTypeOf<[Index, ModelicaInteger]>>:$indices);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "Value":$source, "ValueRange":$indices), [{
            auto slicedArrayType = source.getType().cast<ArrayType>().slice(indices.size());
            build($_builder, $_state, slicedArrayType, source, indices);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getSourceArrayType() {
            return getSource().getType().cast<ArrayType>();
        }

        ArrayType getResultArrayType() {
            return getResult().getType().cast<ArrayType>();
        }

        //===------------------------------------------------------------------===//
        // ViewLikeOpInterface methods
        //===------------------------------------------------------------------===//

        Value getViewSource() {
            return getSource();
        }
    }];
}

def Modelica_ArrayFromElementsOp : Modelica_Op<"array_from_elements",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Create an array with the given elements.";

    let description = [{
        Create an array with the given elements.
        The amount of operands must match the total number of values of the array.

        Example:

        ```mlir
        %0 = constant #modelica.int<1>
        %1 = constant #modelica.real<2.0>
        %2 = constant #modelica.real<3.0>
        %3 = constant #modelica.int<4>
        %4 = constant #modelica.int<5>
        %5 = constant #modelica.int<6>
        %1 = modelica.array_from_elements %0, %1, %2, %3, %4, %5 : !modelica.int, !modelica.real, !modelica.real, !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $values attr-dict `:` qualified(type($values)) `->` qualified(type($array))
    }];

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$values), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(values);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Member operations
//===----------------------------------------------------------------------===//

def Modelica_MemberCreateOp : Modelica_Op<"member_create", [Symbol]>
{
    let summary = "Declare a member of a class.";

    let description = [{
        The operation is used to declare a member of a class.
        It takes the name of the symbol, its type and optional values
        indicating the sizes of the unknown dimensions, in a way similar
        to `modelica.alloca` and `modelica.alloc`.
        The member value can be then read and written by means of the
        `modelica.member_load` and `modelica.member_store` operations.

        Example:

        ```mlir
        %0 = modelica.member_create @x : !modelica.member<3x5x4x!modelica.int>
        ```
    }];

    let arguments = (ins StrAttr:$sym_name, Variadic<Index>:$dynamicSizes);
    let results = (outs AnyMember:$member);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$name, "MemberType":$memberType, "ValueRange":$dynamicSizes), [{
            auto nameAttr = $_builder.getStringAttr(name);
            build($_builder, $_state, memberType, nameAttr, dynamicSizes);
    }]>];

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return getMember().getType().cast<MemberType>();
        }

        bool isInput() {
            return getMemberType().isInput();
        }

        bool isOutput() {
            return getMemberType().isOutput();
        }

        bool isConstant() {
            return getMemberType().isConstant();
        }
    }];
}

def Modelica_MemberLoadOp : Modelica_Op<"member_load",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Load the data held by a class member.";

    let description = [{
        Load the data held by member of a class.
    }];

    let arguments = (ins AnyMember:$member);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "Value":$member), [{
            auto resultType = member.getType().cast<MemberType>().unwrap();
            build($_builder, $_state, resultType, member);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return getMember().getType().cast<MemberType>();
        }
    }];
}

def Modelica_MemberStoreOp : Modelica_Op<"member_store",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Store data into a class member.";

    let description = [{
        Store data into the member of a class.
    }];

    let arguments = (ins AnyMember:$member, AnyType:$value);
    let results = (outs);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return getMember().getType().cast<MemberType>();
        }
    }];
}

def Modelica_StartOp : Modelica_Op<"start",
    [ParentOneOf<["ModelOp"]>,
    SingleBlockImplicitTerminator<"YieldOp">]>
{
    let summary = "Start attribute of a variable";

    let description = [{
        The operation describes how to compute the initial value of a variable.
    }];

    let arguments = (ins AnyType:$variable, BoolAttr:$fixed, BoolAttr:$each);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

def Modelica_ConstantOp : Modelica_Op<"constant",
    [ConstantLike,
    NoMemoryEffect,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Constant.";

    let description = [{
        The operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute.

        Example:

        ```mlir
        %0 = modelica.constant #modelica.int<3>
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Attribute":$value), [{
            build($_builder, $_state, value.cast<mlir::TypedAttr>().getType(), value);
        }]>
    ];

    let hasFolder = 1;
}

def Modelica_NegateOp : Modelica_Op<"neg",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Negation operation.";

    let description = [{
        The operation takes one argument and returns the value with
        the opposite sign.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AddOp : Modelica_Op<"add",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Addition.";

    let description = [{
        The operation computes the sum of two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the Modelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AddEWOp : Modelica_Op<"add_ew",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise addition.";

    let description = [{
        The operation computes the element-wise sum of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.real
        %2 = modelica.add_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SubOp : Modelica_Op<"sub",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Subtraction.";

    let description = [{
        The operation computes the difference between two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the Modelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.sub %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SubEWOp : Modelica_Op<"sub_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise subtraction.";

    let description = [{
        The operation computes the element-wise difference of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.real
        %2 = modelica.sub_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_MulOp : Modelica_Op<"mul",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Multiplication.";

    let description = [{
        The operation computes the product of two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the Modelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.int>
        %1 = ... : !modelica.array<3x!modelica.int>
        %2 = modelica.mul %0, %1 : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_MulEWOp : Modelica_Op<"mul_ew",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise multiplication.";

    let description = [{
        The operation computes the element-wise product of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.int
        %2 = modelica.mul_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.int) -> !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_DivOp : Modelica_Op<"div",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Division.";

    let description = [{
        The operation computes the division between two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the Modelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.div %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_DivEWOp : Modelica_Op<"div_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise division.";

    let description = [{
        The operation computes the element-wise division of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.int
        %2 = modelica.div_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.int) -> !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_PowOp : Modelica_Op<"pow",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Pow.";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` qualified(type($base)) `,` qualified(type($exponent)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_PowEWOp : Modelica_Op<"pow_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise pow.";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` qualified(type($base)) `,` qualified(type($exponent)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

// Base class for comparison operations.
class Modelica_ComparisonOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits>
{
    let arguments = (ins
        AnyModelicaScalar:$lhs,
        AnyModelicaScalar:$rhs);

    let results = (outs AnyModelicaScalar:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];
}

def Modelica_EqOp : Modelica_ComparisonOp<"eq", [Commutative]>
{
    let summary = "Check if two values are equal.";

    let description = [{
        The operation checks whether two scalar values are equal.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.eq %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

def Modelica_NotEqOp : Modelica_ComparisonOp<"neq", [Commutative]>
{
    let summary = "Check if two values are not equal.";

    let description = [{
        The operation checks whether two scalar values are not equal.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.neq %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

def Modelica_GtOp : Modelica_ComparisonOp<"gt", []>
{
    let summary = "Check if a value is greater than another one.";

    let description = [{
        The  operation checks whether a scalar value is greater than another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.gt %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

def Modelica_GteOp : Modelica_ComparisonOp<"gte", []>
{
    let summary = "Check if a value is greater or equal than another one.";

    let description = [{
        The  operation checks whether a scalar value is greater or equal to another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.gte %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

def Modelica_LtOp : Modelica_ComparisonOp<"lt", []>
{
    let summary = "Check if a value is less than another one.";

    let description = [{
        The operation checks whether a scalar value is less than another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.lt %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

def Modelica_LteOp : Modelica_ComparisonOp<"lte", []>
{
    let summary = "Check if a value is less or equal than another one.";

    let description = [{
        The operation checks whether a scalar value is less or equal to another one.

        Example:
        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.lte %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

def Modelica_NotOp : Modelica_Op<"not",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Compute the logical NOT of a boolean value.";

    let description = [{
        The operation computes the logical NOT of a boolean value.
        It takes one operand and returns one result. The type of the operand must be a
        Modelica boolean or an array composed by booleans. In case of array operand, its
        size must be equal to the one of the result type. Runtime checks can also be
        enabled to ensure shapes correctness when dynamic dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = modelica.not %0 : !modelica.bool -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = modelica.not %0, %1 : !modelica.array<3x!modelica.bool> -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AndOp : Modelica_Op<"and",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Compute the logical AND between two boolean values.";

    let description = [{
        The operation computes the logical AND between two boolean values.
        It takes two operands and returns one result. The type of the operands must be
        Modelica booleans or arrays composed by booleans. In case of array operands, their
        size must be equal among themselves and also with respect to the result type.
        Runtime checks can also be enabled to ensure shapes correctness when dynamic
        dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = ... : !modelica.bool
        %2 modelica.and %0, %1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = ... : !modelica.array<3x!modelica.bool>
        %2 = modelica.and %0, %1 : (!modelica.array<3x!modelica.bool>, !modelica.array<3x!modelica.bool>) -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_OrOp : Modelica_Op<"or",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Compute the logical OR between two boolean values.";

    let description = [{
        The operation computes the logical OR between two boolean values.
        It takes two operands and returns one result. The type of the operands must be
        Modelica booleans or arrays composed by booleans. In case of array operands, their
        size must be equal among themselves and also with respect to the result type.
        Runtime checks can also be enabled to ensure shapes correctness when dynamic
        dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = ... : !modelica.bool
        %2 = modelica.or %0, %1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = ... : !modelica.array<3x!modelica.bool>
        %2 = modelica.or %0, %1 : (!modelica.array<3x!modelica.bool>, !modelica.array<3x!modelica.bool>) -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Built-in operations
//===----------------------------------------------------------------------===//

def Modelica_AbsOp : Modelica_Op<"abs",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
    let summary = "Compute the absolute value.";

    let description = [{
        The operation computes the absolute value of another one.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.abs %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AcosOp : Modelica_Op<"acos",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric inverse cosine function.";

    let description = [{
        The operation computes the inverse cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.acos %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AsinOp : Modelica_Op<"asin",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric inverse sine function";

    let description = [{
        The operation computes the inverse sine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.asin %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_AtanOp : Modelica_Op<"atan",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric inverse tangent function.";

    let description = [{
        The operation computes the inverse tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.atan %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_Atan2Op : Modelica_Op<"atan2",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Four quadrant inverse tangent function.";

    let description = [{
        The operation takes two values y and x and returns the inverse tangent of y/x
        Usually y and x provided in such a form that if v = atan2(y, x), then y = sin(v) and x = cos(v).

        Example:

        ```mlir
        %y = ... : !modelica.real
        %x = ... : !modelica.real
        %v = modelica.atan2 %y, %x : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$y, AnyType:$x);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $y `,` $x attr-dict `:` `(` qualified(type($y)) `,` qualified(type($x)) `)` `->` qualified(type($result))
    }];
}

def Modelica_CeilOp : Modelica_Op<"ceil",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
  let summary = "Ceil function.";

  let description = [{
    The function takes an operand x and returns the smallest integer not less than x.

    Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.ceil %x : !modelica.real -> !modelica.real
        ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];

  let hasFolder = 1;
}

def Modelica_CosOp : Modelica_Op<"cos",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric cosine function.";

    let description = [{
        The operation computes the cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.cos %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_CoshOp : Modelica_Op<"cosh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Hyperbolic cosine function.";

    let description = [{
        The operation computes the hyperbolic cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.cosh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_DiagonalOp : Modelica_Op<"diagonal",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get a diagonal matrix with the given values";

    let description = [{
        Given an array of values, the operation returns a matrix
        having those elements on the main diagonal.

        Example:

        ```mlir
        $0 = ... : !modelica.array<3x!modelica.real>
        %1 = modelica.diagonal %0 : !modelica.array<3x!modelica.real> -> !modelica.array<3x3x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$values);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $values attr-dict `:` qualified(type($values)) `->` qualified(type($result))
    }];
}

def Modelica_DivTruncOp : Modelica_Op<"div_trunc", []>
{
    let summary = "Get the integer part of a quotient.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns
        the algebraic quotient `x / y` with any fractional part discarded.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.div_trunc %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` qualified(type($x)) `,` qualified(type($y)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_ExpOp : Modelica_Op<"exp",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Base 'e' exponential function.";

    let description = [{
        The function takes an operand x and returns the e^x value.

        Example:

        ```mlir
        %exponent = ... : !modelica.real
        %0 = modelica.exp %exponent : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $exponent attr-dict `:` qualified(type($exponent)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_FloorOp : Modelica_Op<"floor",
   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
  let summary = "Floor function.";

  let description = [{
    The function takes an operand x and returns the largest integer not greater than x.

    Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.floor %x : !modelica.real -> !modelica.real
        ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];

  let hasFolder = 1;
}

def Modelica_IdentityOp : Modelica_Op<"identity",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Returns an identity matrix with a given dimensions size.";

    let description = [{
        The operation takes one operand, representing the matrix size,
        and returns an identity matrix.

        Example:

        ```mlir
        %dim = constant 3 : index
        %identity = modelica.identity %dim : index -> !modelica.array<?x?x!Modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$size);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $size attr-dict `:` qualified(type($size)) `->` qualified(type($result))
    }];
}

def Modelica_IntegerOp : Modelica_Op<"integer",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
  let summary = "Integer function.";

  let description = [{
    The function takes an operand x and returns the largest integer not greater than x.

    Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.integer %x : !modelica.real -> !modelica.int
        ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
  }];

  let hasFolder = 1;
}

def Modelica_LinspaceOp : Modelica_Op<"linspace",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get an array with equally spaced values.";

    let description = [{
        The operation creates an array with equally spaces values.
        It takes three operands representing the first value that will be present
        in the array, the last value, and the number of elements that will populate
        the array. All the elements of the resulting array will be equally spaced
        according to the desired amount of values.
    }];

    let arguments = (ins AnyType:$begin, AnyType:$end, AnyType:$amount);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $begin `,` $end `,` $amount attr-dict `:` `(` qualified(type($begin)) `,` qualified(type($end)) `,` qualified(type($amount)) `)` `->` qualified(type($result))
    }];
}

def Modelica_LogOp : Modelica_Op<"log",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Natural (base e) logarithm function.";

    let description = [{
        The operation computes the natural (base 'e') logarithm of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.log %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_Log10Op : Modelica_Op<"log10",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Base 10 logarithm function.";

    let description = [{
        The operation computes the base 10 logarithm of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.log10 %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_MaxOp : Modelica_Op<"max",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the greatest value among an array or two scalar values.";

    let description = [{
        Get the greatest value among an array or two scalar values.

        Example with array value:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.max %0 : !modelica.array<5x!modelica.int> -> !modelica.int
        ```

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.max %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$first, Optional<AnyType>:$second);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "Type":$result, "ValueRange":$values), [{
            $_state.addTypes(result);
            $_state.addOperands(values);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
}

def Modelica_MinOp : Modelica_Op<"min",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the smallest value among an array or two scalar values.";

    let description = [{
        Get the smallest value among an array or two scalar values.

        Example with array value:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.min %0 : !modelica.array<5x!modelica.int> -> !modelica.int
        ```

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.min %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$first, Optional<AnyType>:$second);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "Type":$result, "ValueRange":$values), [{
            $_state.addTypes(result);
            $_state.addOperands(values);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
}

def Modelica_ModOp : Modelica_Op<"mod", []>
{
    let summary = "Get the modulus of a division of two numbers.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns the integer
        modulus of `x / y`, i.e., `mod(x, y) = x - floor(x / y) * y`.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.mod %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` qualified(type($x)) `,` qualified(type($y)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_NDimsOp : Modelica_Op<"ndims", []>
{
    let summary = "Get the number of dimensions of an array.";

    let description = [{
        The operation takes an array as argument and returns
        the number of its dimensions.

        Example:

        ```mlir
        %0 = ... : !modelica.array<4x?x5x!modelica.int>
        %1 = modelica.ndims %0 : !modelica.array<4x?x5x!modelica.int> -> !modelica.int
        // %1 will have value 3.
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs AnyModelicaScalar:$result);

    let assemblyFormat = [{
        $array attr-dict `:` qualified(type($array)) `->` qualified(type($result))
    }];
}

def Modelica_OnesOp : Modelica_Op<"ones",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get an array with all the elements set to '1'";

    let description = [{
        Returns an array with "1" elements.
        The dimensions of the array are specified as input.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = constant 3 : index
        %2 = constant 4 : index

        %array = modelica.ones %0, %1, %2 : (index, index, index) -> !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$sizes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` `(` qualified(type($sizes)) `)` `->` qualified(type($result))
    }];
}

def Modelica_ProductOp : Modelica_Op<"product", []>
{
    let summary = "Get the product of the values of an array.";

    let description = [{
        The operation takes an array and computes the product of all its values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.real>
        %1 = modelica.product %0 : !modelica.array<?x?x!modelica.real> -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` qualified(type($array)) `->` qualified(type($result))
    }];
}

def Modelica_RemOp : Modelica_Op<"rem", []>
{
    let summary = "Get the modulus of a division of two numbers.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns the integer
        remainder of `x / y`, such that `div(x,y) * y + rem(x, y) = x`.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = ... : !modelica.real
        %2 = modelica.rem %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` qualified(type($x)) `,` qualified(type($y)) `)` `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SignOp : Modelica_Op<"sign",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
    let summary = "Get an integer representing the sign of a value.";

    let description = [{
        The operation takes an operand and returns 1 if its value is greater
        than zero, -1 if less than zero or 0 if equal to zero.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sign %0 : !modelica.real -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SinOp : Modelica_Op<"sin",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric sine function.";

    let description = [{
        The operation computes the sine of a value.
        It takes one operand and returns one result. If the operand is an array,
        then the Modelica vectorization rules are applied and an array with the
        same dimensions is returned.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SinhOp : Modelica_Op<"sinh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Hyperbolic sine function.";

    let description = [{
        The operation computes the hyperbolic sine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sinh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SizeOp : Modelica_Op<"size",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the dimensions of an array.";

    let description = [{
        The operation can accept either one or two arguments.

        In the first case the operand represents the array of interest and
        the result consists in an array containing the size of its dimensions.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.int>
        %1 = modelica.size %0 : !modelica.array<?x?x!modelica.int> -> !modelica.array<2x!modelica.int>
        ```

        In the second case the second operand also specifies the (zero-based)
        dimension of interest, and thus the operation returns just a scalar
        value corresponding to the size of that dimension.

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.int>
        %1 = constant 1 : index
        %2 = modelica.size %0, %1 : (!modelica.array<?x?x!modelica.int>, index) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$array, Optional<AnyType>:$dimension);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        bool hasDimension() {
            return getNumOperands() == 2;
        }
    }];
}

def Modelica_SqrtOp : Modelica_Op<"sqrt",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Square root.";

    let description = [{
        The operation computes the square root of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sqrt %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_SumOp : Modelica_Op<"sum", []>
{
    let summary = "Get the sum of the values of an array.";

    let description = [{
        The operation takes an array and computes the sum of all its values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.real>
        %1 = modelica.sum %0 : !modelica.array<?x?x!modelica.real> -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` qualified(type($array)) `->` qualified(type($result))
    }];
}

def Modelica_SymmetricOp : Modelica_Op<"symmetric",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get a symmetric matrix.";

    let description = [{
        The operation takes a matrix and returns a new symmetric ones that is built
        by considering only the elements above the source diagonal and mirroring
        them into the bottom part.

        Example:

        ```
        %0 = ... : !modelica.array<3x3x!modelica.int>
        %1 = modelica.symmetric %0 : !modelica.array<3x3x!modelica.int> -> !modelica.array<3x3x!modelica.int>

        //      [ 1 2 3 ]           [ 1 2 3 ]
        // %0 = [ 4 5 6 ]  ->  %1 = [ 2 5 6 ]
        //      [ 7 8 9 ]           [ 3 6 9 ]
        ```
    }];

    let arguments = (ins AnyType:$matrix);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $matrix attr-dict `:` qualified(type($matrix)) `->` qualified(type($result))
    }];
}

def Modelica_TanOp : Modelica_Op<"tan",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Trigonometric tangent function.";

    let description = [{
        The operation computes the tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.tan %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_TanhOp : Modelica_Op<"tanh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Hyperbolic tangent function.";

    let description = [{
        The operation computes the hyperbolic tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.tanh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];

    let hasFolder = 1;
}

def Modelica_TransposeOp : Modelica_Op<"transpose",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Matrix transposition.";

    let description = [{
        The operation takes a matrix and computes its transpose.

        Example:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.transpose : !modelica.array<3x2x!modelica.int> -> !modelica.array<2x3x!modelica.int>

        //      [ 1 2 ]
        // %0 = [ 3 4 ]  ->  %1 = [ 1 3 5 ]
        //      [ 5 6 ]           [ 2 4 6 ]
        ```
    }];

    let arguments = (ins AnyType:$matrix);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $matrix attr-dict `:` qualified(type($matrix)) `->` qualified(type($result))
    }];
}

def Modelica_ZerosOp : Modelica_Op<"zeros",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get an array with all the elements set to '0'";

    let description = [{
        Returns an array with "0" elements.
        The dimensions of the array are specified as input.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = constant 3 : index
        %2 = constant 4 : index

        %array = modelica.zeros %0, %1, %2 : (index, index, index) -> !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$sizes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` `(` qualified(type($sizes)) `)` `->` qualified(type($result))
    }];
}

//===----------------------------------------------------------------------===//
// Modeling operations
//===----------------------------------------------------------------------===//

def Modelica_DerOp : Modelica_Op<"der", []>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` qualified(type($operand)) `->` qualified(type($result))
    }];
}

def Modelica_ModelOp : Modelica_Op<"model",
    [IsolatedFromAbove,
    Symbol,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    NoTerminator]>
{
    let summary = "Modelica model.";

    let description = [{
        The operation represents a Modelica model.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        AnyRegion:$varsRegion,
        AnyRegion:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        SmallVector<StringRef> variableNames();

        Block* bodyBlock();
    }];
}

def Modelica_ForEquationOp : Modelica_Op<"for_equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    NoTerminator]>
{
    let summary = "Loop defining more equation through a for loop.";

    let description = [{
        The for equation loop is an operation used to declare multiple
        equations whose structure just differ by the indices they depend
        on. In this sense, the loop is not the standard algorithmic loop
        developers are used to, but rather a modeling construct that
        changes the model structure.
        The iteration starts from the `from` value (included) and ends
        with the `to` value (excluded), with a increment of 'step'.
        Multiple for loops can be nested into each other.

        Example:

        ```mlir
        modelica.for_equation {from = 3, to = 9, step = 2} {
            modelica.equation {
                ...
            }
        }
        ```
    }];

    let arguments = (ins IndexAttr:$from, IndexAttr:$to, IndexAttr:$step);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "long":$from, "long":$to, "long":$step)>
    ];

    let extraClassDeclaration = [{
        Block* bodyBlock();
        Value induction();
    }];
}

def Modelica_EquationOp : Modelica_Op<"equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    EquationInterface]>
{
    let summary = "Equation.";

    let description = [{
        The operation represents a Modelica equation.
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Block* bodyBlock();
    }];
}

def Modelica_InitialEquationOp : Modelica_Op<"initial_equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    EquationInterface]>
{
    let summary = "Initial equation.";

    let description = [{
        The operation represents a Modelica initial equation.
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Block* bodyBlock();
    }];
}

def Modelica_AlgorithmOp : Modelica_Op<"algorithm",
    [ParentOneOf<["ModelOp"]>,
    AlgorithmInterface,
    NoTerminator]>
{
  let summary = "Algorithm.";

  let description = [{
      The operation represents a Modelica algorithm.
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region
      SizedRegion<1>:$bodyRegion
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Block* bodyBlock();
  }];
}

def Modelica_InitialAlgorithmOp : Modelica_Op<"initial_algorithm",
    [ParentOneOf<["ModelOp"]>,
    AlgorithmInterface,
    NoTerminator]>
{
  let summary = "Initial algorithm.";

  let description = [{
      The operation represents a Modelica initial algorithm.
  }];

  let arguments = (ins);
  let results = (outs);

  let regions = (region
      SizedRegion<1>:$bodyRegion
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    Block* bodyBlock();
  }];
}

def Modelica_TimeOp : Modelica_Op<"time",
    [ParentOneOf<["EquationOp", "InitialEquationOp"]>]>
{
    let summary = "Time.";

    let description = [{
        The operation represents the time of the simulation.
    }];

    let arguments = (ins);
    let results = (outs ModelicaReal);

    let assemblyFormat = [{
        attr-dict `:` qualified(type(results))
    }];

    let builders = [
        OpBuilder<(ins), [{
            auto realType = RealType::get($_builder.getContext());
            build($_builder, $_state, realType);
        }]>
    ];
}

def Modelica_SelectOp : Modelica_Op<"select", [SameVariadicOperandSize]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        AnyType:$condition,
        Variadic<AnyType>:$trueValues,
        Variadic<AnyType>:$falseValues);

    let results = (outs Variadic<AnyType>:$results);

    let hasCustomAssemblyFormat = 1;
}

def Modelica_EquationSideOp : Modelica_Op<"equation_side",
    [ParentOneOf<["EquationOp", "InitialEquationOp"]>]>
{
    let summary = "Equation side.";

    let description = [{
        The operation represents one of the sides (left or right)
        of an equation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyTuple);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "ValueRange":$values), [{
            auto tupleType = $_builder.getTupleType(values.getTypes());
            build($_builder, $_state, tupleType, values);
        }]>
    ];
}

def Modelica_EquationSidesOp : Modelica_Op<"equation_sides",
    [ParentOneOf<["EquationOp", "InitialEquationOp"]>,
    Terminator]>
{
    let summary = "Equation.";

    let description = [{
        The operation represents a Modelica equation.
    }];

    let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
    let results = (outs);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs))
    }];

    let extraClassDeclaration = [{
        mlir::ValueRange getLhsValues() {
            auto sideOp = getLhs().getDefiningOp();
            assert(sideOp != nullptr && mlir::isa<EquationSideOp>(sideOp));
            return mlir::cast<EquationSideOp>(sideOp).getValues();
        }

        mlir::ValueRange getRhsValues() {
            auto sideOp = getRhs().getDefiningOp();
            assert(sideOp != nullptr && mlir::isa<EquationSideOp>(sideOp));
            return mlir::cast<EquationSideOp>(sideOp).getValues();
        }
    }];
}

def Modelica_FunctionOp : Modelica_Op<"function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica function.";

    let description = [{
        The operation represents a Modelica function.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type);

    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        mlir::Block* bodyBlock();

        SmallVector<StringRef> inputMemberNames();
        SmallVector<StringRef> outputMemberNames();
        SmallVector<StringRef> protectedMemberNames();

        bool shouldBeInlined();

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        Region *getCallableRegion() {
            return isDeclaration() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return false;
        }
    }];
}

def Modelica_RawFunctionOp : Modelica_Op<"raw_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    FunctionOpInterface,
    CallableOpInterface]>
{
    let summary = "Raw function.";

    let description = [{
        TODO
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility);

    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name, "mlir::FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        bool shouldBeInlined();

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            mlir::Operation::dialect_attr_range attrs);

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            llvm::ArrayRef<mlir::NamedAttribute> attrs,
            llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

        /// Create a deep copy of this function and all of its blocks, remapping any
        /// operands that use values outside of the function using the map that is
        /// provided (leaving them alone if no entry is present). If the mapper
        /// contains entries for function arguments, these arguments are not
        /// included in the new function. Replaces references to cloned sub-values
        /// with the corresponding value that is copied, and adds those mappings to
        /// the mapper.
        RawFunctionOp clone(mlir::BlockAndValueMapping &mapper);
        RawFunctionOp clone();

        /// Clone the internal blocks and attributes from this function into dest.
        /// Any cloned blocks are appended to the back of dest. This function
        /// asserts that the attributes of the current function and dest are
        /// compatible.
        void cloneInto(RawFunctionOp dest, mlir::BlockAndValueMapping &mapper);

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion() {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return isExternal();
        }
    }];
}

def RawReturnOp : Modelica_Op<"raw_return",
    [NoMemoryEffect,
    HasParent<"RawFunctionOp">,
    ReturnLike, Terminator]>
{
  let summary = "Raw function return operation.";

  let description = [{
    The `modelica.return` operation represents a return operation within a raw
    function. The operation takes variable number of operands and produces no
    results. The operand number and types must match the signature of the
    function that contains the operation.

    Example:

    ```mlir
    modelica.raw_func @foo() : (!modelica.int, !modelica.real) {
      ...
      modelica.raw_return %0, %1 : !modelica.int, !modelica.real
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, llvm::None);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def Modelica_RuntimeFunctionOp : Modelica_Op<"runtime_function",
    [Symbol, CallableOpInterface]>
{
    let summary = "Runtime function.";

    let description = [{
        A Modelica function that is implemented into the runtime library.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility);

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name, "mlir::FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion() {
            return nullptr;
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return true;
        }
    }];
}

def Modelica_CallOp : Modelica_Op<"call",
    [CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>]>
{
    let summary = "Call operation.";

    let description = [{
        The operation represents a direct call to a user-defined Modelica function
        that is within the same symbol scope as the call. The operands and result
        types of the call must match the specified function type. The callee is
        encoded as a symbol reference attribute named "callee".

        Example:

        ```mlir
        %2 = modelica.call @foo(%0, %1) : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args, OptionalAttr<AnyI64Attr>:$movedResults);
    let results = (outs Variadic<AnyType>);

    let assemblyFormat = [{
        $callee `(` $args `)` attr-dict `:` functional-type($args, results)
    }];

    let builders = [
        OpBuilder<(ins "FunctionOp":$callee, CArg<"mlir::ValueRange", "{}">:$args), [{
            $_state.addOperands(args);
            $_state.addAttribute("callee", SymbolRefAttr::get($_builder.getContext(), callee.getSymName()));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "RawFunctionOp":$callee, CArg<"mlir::ValueRange", "{}">:$args), [{
            $_state.addOperands(args);
            $_state.addAttribute("callee", SymbolRefAttr::get($_builder.getContext(), callee.getSymName()));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "RuntimeFunctionOp":$callee, CArg<"mlir::ValueRange", "{}">:$args), [{
            $_state.addOperands(args);
            $_state.addAttribute("callee", SymbolRefAttr::get($_builder.getContext(), callee.getSymName()));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "mlir::SymbolRefAttr":$callee, "mlir::TypeRange":$results, CArg<"mlir::ValueRange", "{}">:$args), [{
            $_state.addOperands(args);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(results);
        }]>,
        OpBuilder<(ins "llvm::StringRef":$callee, "mlir::TypeRange":$results, CArg<"mlir::ValueRange", "{}">:$args), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), results, args);
        }]>
    ];

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallOpInterface
        //===------------------------------------------------------------------===//

        mlir::FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return { arg_operand_begin(), arg_operand_end() };
        }

        operand_iterator arg_operand_begin() {
            return operand_begin();
        }

        operand_iterator arg_operand_end() {
            return operand_end();
        }

        /// Return the callee of this operation.
        mlir::CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }
    }];
}



def Modelica_AssignmentOp : Modelica_Op<"assignment",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$destination, AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $destination `,` $value attr-dict `:` qualified(type($destination)) `,` qualified(type($value))
    }];
}

def Modelica_DerFunctionOp : Modelica_Op<"der_function",
    [IsolatedFromAbove, Symbol, CallableOpInterface]>
{
    let summary = "Modelica derivative function.";

    let description = [{
        The operation represents the derivative of a Modelica function.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolNameAttr:$derived_function,
        StrArrayAttr:$independent_vars);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        Region* getCallableRegion() {
            // The function body will be created by an appropriate pass, and at the same
            // type the der function will be converted to a standard function. Thus,
            // this operation is like an external function.

            return nullptr;
        }

        /// Returns the results types that the callable region produces when executed.
        ArrayRef<Type> getCallableResults();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return false;
        }
    }];
}

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Modelica_BreakOp : Modelica_Op<"break", [Terminator]>
{
    let summary = "Loop breaking operation.";

    let description = [{
        The `modelica.break` operation terminates the execution of the closest
        loop that contains the instruction.

        Example:

        ```mlir
        modelica.while {
            ...
        } do {
            %condition = ... : !modelica.bool

            modelica.if %condition {
                modelica.break
            }

            modelica.yield
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let assemblyFormat = [{
        attr-dict
    }];
}

def Modelica_ConditionOp : Modelica_Op<"condition",
    [Terminator,
    ParentOneOf<["ForOp", "WhileOp"]>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins ModelicaBoolean:$condition, Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        `(` $condition `:` qualified(type($condition)) `)` attr-dict ($values^ `:` qualified(type($values)))?
    }];

    let builders = [
        OpBuilder<(ins "::mlir::Value":$condition), [{
            build($_builder, $_state, condition, llvm::None);
        }]>
    ];
}

def Modelica_ForOp : Modelica_Op<"for",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "For algorithmic loop.";

    let description = [{
        TODO
    }];

    let arguments = (ins Variadic<AnyType>:$args);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion,
        SizedRegion<1>:$stepRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        Block* conditionBlock();
        Block* bodyBlock();
        Block* stepBlock();
    }];
}

def Modelica_IfOp : Modelica_Op<"if",
    [NoTerminator,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Algorithmic if-else construct.";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$condition);
    let results = (outs);

    let regions = (region
      SizedRegion<1>:$thenRegion,
      AnyRegion:$elseRegion
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$condition, "bool":$withElseRegion)>
    ];

    let extraClassDeclaration = [{
        Block* thenBlock();
        Block* elseBlock();
    }];
}

def Modelica_ReturnOp : Modelica_Op<"return", [Terminator]>
{
    let summary = "Early termination instruction for functions.";

    let description = [{
        The `modelica.return` operation terminates the execution of the
        whole function, making it returning the values as computed so far.

        Example:

        ```mlir
        modelica.function @foo ... {
            %condition = ... : !modelica.bool

            modelica.if %condition {
                modelica.return
            }
            ...
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let assemblyFormat = [{
        attr-dict
    }];
}

def Modelica_WhileOp : Modelica_Op<"while",
    [NoTerminator,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "While algorithmic loop.";

    let description = [{
        TODO
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
}

def Modelica_YieldOp : Modelica_Op<"yield",
    [Terminator,
    ParentOneOf<["ForEquationOp", "IfOp", "ForOp", "WhileOp", "ModelOp", "StartOp"]>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        attr-dict ($values^ `:` qualified(type($values)))?
    }];
}

//===----------------------------------------------------------------------===//
// Utility operations
//===----------------------------------------------------------------------===//

def Modelica_CastOp : Modelica_Op<"cast",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Cast a scalar value to another type.";

    let description = [{
        Cast a value from a Modelica-compatible type to another Modelica-compatible
        type. The operation does allow only for scalars to be casted. Also the `index`
        type can be used for casts.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.cast %0 : !modelica.int -> !modelica.real
        ```
    }];

    let arguments = (ins AnyModelicaScalar:$value);
    let results = (outs AnyModelicaScalar:$result);

    let assemblyFormat = [{
        $value attr-dict `:` qualified(type($value)) `->` qualified(type($result))
    }];
}

def Modelica_PrintOp : Modelica_Op<"print",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Print a value to the standard output.";

    let description = [{
        The operation prints a value to the standard output.
        It is thought for debugging purpose, especially for tests.
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $value attr-dict `:` qualified(type($value))
    }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
