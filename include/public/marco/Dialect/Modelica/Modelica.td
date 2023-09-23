#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "marco/Dialect/Modelica/ModelicaAttributes.td"
include "marco/Dialect/Modelica/ModelicaDialect.td"
include "marco/Dialect/Modelica/ModelicaOpInterfaces.td"
include "marco/Dialect/Modelica/ModelicaTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

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

// Modelica Record type
def ModelicaRecord : Type<CPred<"$_self.isa<::mlir::modelica::RecordType>()">,
                        "real", "::mlir::modelica::RecordType">;
// Modelica scalar type.
def AnyModelicaScalar : AnyTypeOf<[
    ModelicaBoolean, ModelicaInteger, ModelicaReal]>;

// Modelica-compatible scalar type.
def AnyModelicaCompatibleScalar : AnyTypeOf<[
    AnyModelicaScalar, AnyInteger, AnyFloat, Index]>;

// Whether a type is an ArrayType.
def IsModelicaArrayTypePred : CPred<"$_self.isa<::mlir::modelica::ArrayType>()">;

// Array with given allowed element types.
class ModelicaArrayOf<list<Type> allowedTypes> :
    ShapedContainerType<
        allowedTypes, IsModelicaArrayTypePred, "array",
        "::mlir::modelica::ArrayType">;

def AnyModelicaArray : ModelicaArrayOf<[AnyModelicaScalar]>;

def AnyModelicaCompatibleArray : ModelicaArrayOf<[AnyModelicaCompatibleScalar]>;

class ArrayOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsModelicaArrayTypePred,
                  "$_self.cast<::mlir::modelica::ArrayType>().getElementType()",
                  "array", "::mlir::modelica::ArrayType">;

def AnyArray : ArrayOf<[
    ModelicaBoolean, ModelicaInteger, ModelicaReal, Index, ModelicaArray,
    ModelicaRecord]>;

// Whether a type is a VariableType.
def IsVariableTypePred : CPred<"$_self.isa<::mlir::modelica::VariableType>()">;

class VariableOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsVariableTypePred,
                  "$_self.cast<::mlir::modelica::VariableType>().getElementType()",
                  "variable", "::mlir::modelica::VariableType">;

def AnyVariable : VariableOf<[ModelicaBoolean, ModelicaInteger, ModelicaReal, Index]>;

// Equation type.
def ModelicaEquation : Type<CPred<
    "$_self.isa<::mlir::modelica::EquationType>()">,
    "equation", "::mlir::modelica::EquationType">;

// Iteration space type.
def ModelicaIterable : Type<CPred<
    "$_self.isa<::mlir::modelica::IterableType>()">,
    "iterable", "::mlir::modelica::IterableType">;

// Multidimensional range attribute.
def MultidimensionalRangeAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::MultidimensionalRangeAttr>()">,
                 "Multidimensional range attribute">
{
  let storageType = [{ ::mlir::modelica::MultidimensionalRangeAttr }];
  let returnType = [{ ::mlir::modelica::MultidimensionalRangeAttr }];
  let convertFromStorage = "$_self";
}

// Index set attribute.
def IndexSetAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::IndexSetAttr>()">,
                 "Index set attribute">
{
  let storageType = [{ ::mlir::modelica::IndexSetAttr }];
  let returnType = [{ ::mlir::modelica::IndexSetAttr }];
  let convertFromStorage = "$_self";
}

def VarDerivativeAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::VarDerivativeAttr>()">,
                 "Variable's derivative attribute">;

// Equation path attribute.
def EquationPathAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::EquationPathAttr>()">,
                 "Equation path attribute">
{
  let storageType = [{ ::mlir::modelica::EquationPathAttr }];
  let returnType = [{ ::mlir::modelica::EquationPathAttr }];
  let convertFromStorage = "$_self";
}

// Equation scheduling direction attribute.
def EquationScheduleDirectionAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::EquationScheduleDirectionAttr>()">,
                 "Equation scheduling direction attribute">
{
  let storageType = [{ ::mlir::modelica::EquationScheduleDirectionAttr }];
  let returnType = [{ ::mlir::modelica::EquationScheduleDirectionAttr }];
  let convertFromStorage = "$_self";
}

// Equation view attribute.
def EquationViewAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::EquationViewAttr>()">,
                 "Equation view attribute">
{
  let storageType = [{ ::mlir::modelica::EquationViewAttr }];
  let returnType = [{ ::mlir::modelica::EquationViewAttr }];
  let convertFromStorage = "$_self";
}

// Equation match attribute.
def EquationMatchAttr
    : Attr<CPred<"$_self.isa<::mlir::modelica::EquationMatchAttr>()">,
                 "Equation view attribute">
{
  let storageType = [{ ::mlir::modelica::EquationMatchAttr }];
  let returnType = [{ ::mlir::modelica::EquationMatchAttr }];
  let convertFromStorage = "$_self";
}

//===---------------------------------------------------------------------===//
// Traits
//===---------------------------------------------------------------------===//

class NonEmptyVariadic<Type type> : Variadic<type>
{
  let minSize = 1;
}

//===---------------------------------------------------------------------===//
// Modelica operation definitions
//===---------------------------------------------------------------------===//

class Modelica_Op<string mnemonic, list<Trait> traits = []> :
    Op<Modelica_Dialect, mnemonic, traits>;

// Base class for unary operations.
class Modelica_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary operations.
class Modelica_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===---------------------------------------------------------------------===//
// Iteration space operations
//===---------------------------------------------------------------------===//

def RangeOp : Modelica_Op<"range", []>
{
    let summary = "Range";

    let description = [{
        The operation represents a mono-dimensional range, with a lower bound,
        an upper bound and a step.

        Example:

        ```mlir
        %0 = modelica.range %0, %1, %2 : (index, index, index) -> !modelica.iterable<index>
        ```
    }];

    let arguments = (ins
        AnyType:$lowerBound,
        AnyType:$upperBound,
        AnyType:$step);

    let results = (outs ModelicaIterable:$result);

    let assemblyFormat = "$lowerBound `,` $upperBound `,` $step attr-dict `:` `(` type($lowerBound) `,` type($upperBound) `,` type($step) `)` `->` type($result)";

    let hasFolder = 1;
}

def UnboundedRangeOp : Modelica_Op<"unbounded_range", []>
{
    let summary = "Unbounded range";

    let description = [{
        The operation represents a mono-dimensional range with unbounded
        bounds.

        Example:

        ```mlir
        %0 = modelica.unbounded_range : !modelica.iterable<index>
        ```
    }];

    let arguments = (ins);
    let results = (outs ModelicaIterable:$result);

    let assemblyFormat = "attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins), [{
            auto iterableType = IterableType::get(
                $_builder.getContext(), $_builder.getIndexType());

            build($_builder, $_state, iterableType);
        }]>
    ];
}

//===---------------------------------------------------------------------===//
// Array operations
//===---------------------------------------------------------------------===//

def Modelica_AllocaOp : Modelica_Op<"alloca",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Allocate an array on the stack.";

    let description = [{
        Allocate an array on the stack.
        The operation takes a variadic number of arguments, each of one representing the
        size of a dynamic dimension within the desired array type. Note that the amount
        of operands must match the number of dynamic dimensions and they all must be of
        `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` type($array)
    }];

    let hasVerifier = 1;

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
}

def Modelica_AllocOp : Modelica_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Allocate an array on the heap.";

    let description = [{
        Allocate an array on the heap.
        The operation takes a variadic number of arguments, each of one representing the
        size of a dynamic dimension within the desired array type. Note that the amount
        of operands must match the number of dynamic dimensions and they all must be of
        `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        ($dynamicSizes^)? attr-dict `:` type($array)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayFromElementsOp : Modelica_Op<"array_from_elements",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Create an array with the given elements.";

    let description = [{
        Alloc an array with the given elements.
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
        $values attr-dict `:` type($values) `->` type($array)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$values), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(values);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayBroadcastOp : Modelica_Op<"array_broadcast",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Create an array with a given value in all positions.";

    let description = [{
        Alloc an array with all the elements equal to a given one.
        Dynamically sized arrays can be created by passing also the dynamic dimensions.

        Example with static dimensions:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x2x!modelica.real>
        ```

        Example with dynamic dimensions:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : index
        %2 = ... : index
        %3 = modelica.array_broadcast %0, %1, %2 : !modelica.int -> !modelica.array<?x2x?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyModelicaCompatibleScalar:$value, Variadic<Index>:$dynamicDimensions);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $value (`,` $dynamicDimensions^)? attr-dict `:` type($value) `->` type($array)
    }];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_FreeOp : Modelica_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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
        $array attr-dict `:` type($array)
    }];
}

def Modelica_ArrayCastOp : Modelica_Op<"array_cast", [ViewLikeOpInterface]>
{
    let summary = "Cast operation for array types.";

    let description = [{
        This operation should be used for only one purpose, that is to generalize the
        dimensions of an array. This can be done by either converting some dimensions
        to unknown ones or by converting the whole array to an unsized one.
        The operation is NOT intended to cast the element type to a different one or to
        specialize the shape of an array with dynamic dimensions.

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
        $source attr-dict `:` type($source) `->` type($result)
    }];

    let extraClassDeclaration = [{
        /// @name ViewLikeOpInterface methods
        /// {

        Value getViewSource() {
            return getSource();
        }

        /// }
    }];
}

def Modelica_DimOp : Modelica_Op<"dim", [NoMemoryEffect]>
{
    let summary = "Get the size of an array dimension.";

    let description = [{
        Get the size of an array dimension. The operation is trivial in case of
        statically shaped arrays, while it is more useful in case of dimensions that are
        unknown at compile-time.
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
        $array `,` $dimension attr-dict `:` type($array)
    }];

    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$array, "Value":$dimension), [{
            auto indexType = $_builder.getIndexType();
            build($_builder, $_state, indexType, array, dimension);
        }]>
    ];
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
        Load a value from an array location given by indices.
        The number of arguments provided within brackets must match the rank of the
        array. The indices must also have an `index` type.

        Example:

        ```mlir
        $array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = modelica.load $array[%0, %1] : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyArray:$array,
        Variadic<AnyTypeOf<[Index, ModelicaInteger]>>:$indices);

    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$array, CArg<"ValueRange", "{}">:$indices), [{
            auto arrayType = array.getType().cast<ArrayType>();
            build($_builder, $_state, arrayType.getElementType(), array, indices);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the array type.
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
        Store a value to an array location given by indices.
        The value stored must have the same type as the elemental type of the array.
        The number of arguments provided within brackets must match the rank of the
        array. The indices must also have an `index` type.

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
        Variadic<AnyTypeOf<[Index, ModelicaInteger]>>:$indices);

    let results = (outs);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        /// Get the array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_SubscriptionOp : Modelica_Op<"subscription",
    [ViewLikeOpInterface,
    NoMemoryEffect,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Create a view over an array.";

    let description = [{
        Create a view over an array.
        The values of the original arrays are not copied and all the modifications
        performed on the subview do reflect on the original array.

        Example:

        ```mlir
        %array = ... : !modelica.array<3x5x4x!modelica.int>
        %c1 : constant 1 : index
        %c2 : constant 2 : index
        %view = modelica.subscription $array[%c1, %c2] : !modelica.array<3x5x4x!modelica.int>
        // %view has type !modelica.array<4x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyArray:$source,
        Variadic<AnyTypeOf<[Index, ModelicaInteger, ModelicaIterable]>>:$indices);

    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$source, "ValueRange":$indices)>
    ];

    let extraClassDeclaration = [{
        static ArrayType inferResultType(
            ArrayType source, mlir::ValueRange indices);

        ArrayType getSourceArrayType() {
            return getSource().getType().cast<ArrayType>();
        }

        ArrayType getResultArrayType() {
            return getResult().getType().cast<ArrayType>();
        }

        /// @name ViewLikeOpInterface methods
        /// {

        Value getViewSource() {
            return getSource();
        }

        /// }
    }];
}

def Modelica_ArrayFillOp : Modelica_Op<"array_fill",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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

    let arguments = (ins AnyArray:$array, AnyModelicaCompatibleScalar:$value);
    let results = (outs);

    let assemblyFormat = [{
        $array `,` $value attr-dict `:` type($array) `,` type($value)
    }];

    let extraClassDeclaration = [{
        /// Get the array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayCopyOp : Modelica_Op<"array_copy",
    [SameOperandsShape,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Copy the elements from the source to the destination array.";

    let description = [{
        The operation copies the elements from the source to the destination array.

        Example:

        ```mlir
        %source = ... : !modelica.array<?x!modelica.int>
        %destination = ... : !modelica.array<?x!modelica.real>
        modelica.array_copy %destination, %source : !modelica.array<?x!modelica.int>, !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins ModelicaArray:$source, ModelicaArray:$destination);
    let results = (outs);

    let assemblyFormat = [{
        $source `,` $destination attr-dict `:` type($source) `,` type($destination)
    }];
}

//===---------------------------------------------------------------------===//
// Variable operations
//===---------------------------------------------------------------------===//

def Modelica_VariableOp : Modelica_Op<"variable",
    [Symbol,
    ParentOneOf<["ModelOp", "FunctionOp", "RecordOp"]>]>
{
    let summary = "Declare a variable of a class.";

    let description = [{
        The operation is used to declare a variable of a class.
        It takes the name of the symbol, its type and optional values indicating the
        sizes of the unknown dimensions, in a way similar to `modelica.alloca` and
        `modelica.alloc`.
        The variable value can be then read and written by means of the
        `modelica.variable_get` and `modelica.variable_set` operations.

        Example:

        ```mlir
        modelica.variable @x : !modelica.variable<?x5x?x?x!modelica.int> [fixed, unbounded, fixed] {
            %0 = ... : index
            %1 = ... : index
            modelica.yield %0, %1 : index, index
        } attributes {...}
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        StrArrayAttr:$dimensionsConstraints
    );

    let results = (outs);

    let regions = (region AnyRegion:$constraintsRegion);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$name, "VariableType":$variableType)>
    ];

    let extraClassDeclaration = [{
        static constexpr llvm::StringLiteral kDimensionConstraintUnbounded = "unbounded";
        static constexpr llvm::StringLiteral kDimensionConstraintFixed = "fixed";

        VariableType getVariableType() {
            return getType().cast<VariableType>();
        }

        bool isInput() {
            return getVariableType().isInput();
        }

        bool isOutput() {
            return getVariableType().isOutput();
        }

        bool isReadOnly() {
            return getVariableType().isReadOnly();
        }

        bool isDiscrete() {
            return getVariableType().isDiscrete();
        }

        bool isParameter() {
            return getVariableType().isParameter();
        }

        bool isConstant() {
            return getVariableType().isConstant();
        }

        /// Get the number of unbounded dimensions among the unknown ones.
        size_t getNumOfUnboundedDimensions();

        /// Get the number of fixed dimensions among the unknown ones.
        size_t getNumOfFixedDimensions();

        /// Get the indices of the variable, if the variable is an array.
        /// Returns None in case of scalar variables.
        IndexSet getIndices();
    }];
}

def Modelica_VariableGetOp : Modelica_Op<"variable_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Load the data held by a class variable.";

    let description = [{
        Load the data held by variable of a class.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $variable attr-dict `:` type($result)
    }];

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp)>
    ];
}

def Modelica_VariableSetOp : Modelica_Op<"variable_set",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Store data into a class variable.";

    let description = [{
        Store data into the variable of a class.
    }];

    let arguments = (ins SymbolNameAttr:$variable, AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $variable `,` $value attr-dict `:` type($value)
    }];

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp, "mlir::Value":$value)>
    ];
}

def Modelica_ComponentGetOp : Modelica_Op<"component_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Get the value of a component of a variable.";

    let description = [{
        Get the value of a component of a variable.

        Example:

        ```mlir
        modelica.record @R {
            modelica.variable @x : !modelica.variable<!modelica.real>
        }

        modelica.function @Foo {
            modelica.variable @r : !modelica.record<"R">
            modelica.algorithm {
                %0 = modelica.variable_get @r : !modelica.record<"R">
                %1 = modelica.component_get %0, @x, !modelica.record<"R">, !modelica.real
                ...
            }
        }
        ```
    }];

    let arguments = (ins AnyType:$variable, SymbolNameAttr:$componentName);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $variable `,` $componentName attr-dict `:` type($variable) `->` type($result)
    }];
}

def Modelica_ComponentSetOp : Modelica_Op<"component_set",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Set the value of a component of a variable.";

    let description = [{
        Set the value of a component of a variable.

        Example:

        ```mlir
        modelica.record @R {
            modelica.variable @x : !modelica.variable<!modelica.real>
        }

        modelica.function @Foo {
            modelica.variable @r : !modelica.record<"R">
            modelica.algorithm {
                %0 = modelica.variable_get @r : !modelica.record<"R">
                %1 = ... : !modelica.real
                modelica.component_set %0, @x, %1 !modelica.record<"R">, !modelica.real
                ...
            }
        }
        ```
    }];

    let arguments = (ins
        AnyType:$variable,
        SymbolNameAttr:$componentName,
        AnyType:$value
    );

    let results = (outs);

    let assemblyFormat = [{
        $variable `,` $componentName `,` $value attr-dict `:` type($variable) `,` type($value)
    }];
}

def Modelica_GlobalVariableOp : Modelica_Op<"global_variable",
    [Symbol,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Declare a global variable.";

    let description = [{
        // TODO
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value
    );

    let results = (outs);

    let assemblyFormat = [{
        $sym_name attr-dict `:` $type
    }];

    let builders = [
        OpBuilder<(ins "mlir::StringAttr":$name, "mlir::TypeAttr":$type)>,
        OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type)>
    ];
}

def Modelica_GlobalVariableGetOp : Modelica_Op<"global_variable_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Load the data held by a global variable.";

    let description = [{
        Load the data held by a global variable.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $variable attr-dict `:` type($result)
    }];

    let builders = [
        OpBuilder<(ins "GlobalVariableOp":$globalVariableOp)>
    ];
}

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

def Modelica_ConstantOp : Modelica_Op<"constant",
    [ConstantLike,
    Pure,
    AllTypesMatch<["value", "result"]>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Constant.";

    let description = [{
        The operation turns a literal into an SSA value. The data is attached to the
        operation as an attribute.

        Example:

        ```mlir
        %0 = modelica.constant #modelica.int<3>
        ```
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$value attr-dict";
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::TypedAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_AddOp : Modelica_Op<"add",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def Modelica_AddEWOp : Modelica_Op<"add_ew",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SubOp : Modelica_Op<"sub",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SubEWOp : Modelica_Op<"sub_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_MulOp : Modelica_Op<"mul",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_MulEWOp : Modelica_Op<"mul_ew",
    [Commutative,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_PowOp : Modelica_Op<"pow",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Pow.";

    let description = [{
        The operation computes the power of a base to a given exponent.
        It takes to operands (base and exponent) and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = ... : !modelica.real
        %2 = modelica.pow %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_PowEWOp : Modelica_Op<"pow_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Element-wise pow.";

    let description = [{
        The operation computes the element-wise power of a (optionally array) base to a
        given exponent.
        It takes to operands (base and exponent) and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.real>
        %1 = ... : !modelica.real
        %2 = modelica.pow %0, %1 : (!modelica.array<?x!modelica.real>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

// Base class for comparison operations.
class Modelica_ComparisonOp<string mnemonic, list<Trait> traits = []> :
    Modelica_Op<mnemonic, traits>
{
    let arguments = (ins
        AnyModelicaCompatibleScalar:$lhs,
        AnyModelicaCompatibleScalar:$rhs);

    let results = (outs AnyModelicaCompatibleScalar:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

def Modelica_EqOp : Modelica_ComparisonOp<"eq", [Commutative, Pure]>
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

def Modelica_NotEqOp : Modelica_ComparisonOp<"neq", [Commutative, Pure]>
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

def Modelica_GtOp : Modelica_ComparisonOp<"gt", [Pure]>
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

def Modelica_GteOp : Modelica_ComparisonOp<"gte", [Pure]>
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

def Modelica_LtOp : Modelica_ComparisonOp<"lt", [Pure]>
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

def Modelica_LteOp : Modelica_ComparisonOp<"lte", [Pure]>
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

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

def Modelica_NotOp : Modelica_Op<"not",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_AndOp : Modelica_Op<"and",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_OrOp : Modelica_Op<"or",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SelectOp : Modelica_Op<"select",
    [SameVariadicOperandSize, AlwaysSpeculatable]>
{
    let summary = "Ternary operator";

    let description = [{
        The operation is used to select among two group of values according to a
        condition.
    }];

    let arguments = (ins
        AnyType:$condition,
        Variadic<AnyType>:$trueValues,
        Variadic<AnyType>:$falseValues);

    let results = (outs Variadic<AnyType>:$results);

    let hasCustomAssemblyFormat = 1;
}

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

def Modelica_AbsOp : Modelica_Op<"abs",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
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
        $operand attr-dict `:` type($operand) `->` type($result)
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_AtanOp : Modelica_Op<"atan",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
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
        $y `,` $x attr-dict `:` `(` type($y) `,` type($x) `)` `->` type($result)
    }];
}

def Modelica_CeilOp : Modelica_Op<"ceil",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AlwaysSpeculatable,
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
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasFolder = 1;
}

def Modelica_CosOp : Modelica_Op<"cos",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_CoshOp : Modelica_Op<"cosh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_DiagonalOp : Modelica_Op<"diagonal",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $values attr-dict `:` type($values) `->` type($result)
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
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_ExpOp : Modelica_Op<"exp",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $exponent attr-dict `:` type($exponent) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_FillOp : Modelica_Op<"fill",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an array with a given value in all positions.";

    let description = [{
        Alloc an array with all the elements equal to a given one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.fill %0 : !modelica.int -> !modelica.array<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins AnyModelicaCompatibleScalar:$value);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($array)
    }];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType() {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_FloorOp : Modelica_Op<"floor",
   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
   AlwaysSpeculatable,
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
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasFolder = 1;
}

def Modelica_IdentityOp : Modelica_Op<"identity",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $size attr-dict `:` type($size) `->` type($result)
    }];
}

def Modelica_IntegerOp : Modelica_Op<"integer",
  [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  AlwaysSpeculatable,
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
    $operand attr-dict `:` type($operand) `->` type($result)
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
        $begin `,` $end `,` $amount attr-dict `:` `(` type($begin) `,` type($end) `,` type($amount) `)` `->` type($result)
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
        $operand attr-dict `:` type($operand) `->` type($result)
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_MaxOp : Modelica_Op<"max",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_NDimsOp : Modelica_Op<"ndims", [Pure]>
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
    let results = (outs AnyModelicaCompatibleScalar:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];
}

def Modelica_OnesOp : Modelica_Op<"ones",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $sizes attr-dict `:` `(` type($sizes) `)` `->` type($result)
    }];
}

def Modelica_ProductOp : Modelica_Op<"product",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $array attr-dict `:` type($array) `->` type($result)
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
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SignOp : Modelica_Op<"sign",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SinOp : Modelica_Op<"sin",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SinhOp : Modelica_Op<"sinh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SizeOp : Modelica_Op<"size",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_SumOp : Modelica_Op<"sum",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $array attr-dict `:` type($array) `->` type($result)
    }];
}

def Modelica_SymmetricOp : Modelica_Op<"symmetric",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $matrix attr-dict `:` type($matrix) `->` type($result)
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_TanhOp : Modelica_Op<"tanh",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable,
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_TransposeOp : Modelica_Op<"transpose",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $matrix attr-dict `:` type($matrix) `->` type($result)
    }];
}

def Modelica_ZerosOp : Modelica_Op<"zeros",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    AlwaysSpeculatable]>
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
        $sizes attr-dict `:` `(` type($sizes) `)` `->` type($result)
    }];
}

def Modelica_ReductionOp : Modelica_Op<"reduction", [SingleBlock]>
{
    let summary = "Iteration space reduction.";

    let description = [{
        Apply a function on a given iteration space.

        Example:

        ```mlir
        %0 = ... : !modelica.iterable<index>
        %1 = ... : !modelica.iterable<index>

        %2 = modelica.reduction "max", %0, %1 inductions = [%i: index, %j: index] {
            %3 = ... : !modelica.real
            modelica.yield %3 : !modelica.real
        } : !modelica.real
        ```
    }];

    let arguments = (ins
        StrAttr:$action,
        Variadic<ModelicaIterable>:$iterables);

    let results = (outs AnyType:$result);

    let regions = (region
        SizedRegion<1>:$expressionRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        mlir::Block* createExpressionBlock(mlir::OpBuilder& builder);

        auto getInductions()
        {
            return getExpressionRegion().getArguments();
        }
    }];
}

//===---------------------------------------------------------------------===//
// Modeling operations
//===---------------------------------------------------------------------===//

def Modelica_PackageOp : Modelica_Op<"package",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica package.";

    let description = [{
        The operation represents a Modelica package.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    // Ensure that the block inside the region exists.
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        Block* bodyBlock();

        auto getVariables()
        {
            return getOps<VariableOp>();
        }
    }];
}

def Modelica_ModelOp : Modelica_Op<"model",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica model.";

    let description = [{
        The operation represents a Modelica model.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DefaultValuedAttr<
            TypedArrayAttrBase<VarDerivativeAttr, "Derivatives map">,
            "std::nullopt">:$derivatives_map);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        auto getVariables()
        {
            return getOps<VariableOp>();
        }

        /// Collect all the variables.
        void collectVariables(llvm::SmallVectorImpl<VariableOp>& variables);

        /// Collect all the initial and non-initial equations of a model.
        void collectEquations(
            llvm::SmallVectorImpl<EquationInstanceOp>& initialEquations,
            llvm::SmallVectorImpl<EquationInstanceOp>& equations);

        /// Collect all the initial and non-initial matched equations of a
        /// model.
        void collectEquations(
            llvm::SmallVectorImpl<MatchedEquationInstanceOp>& initialEquations,
            llvm::SmallVectorImpl<MatchedEquationInstanceOp>& equations);

        /// Collect all the initial and non-initial SCCs of a model.
        void collectSCCs(
            llvm::SmallVectorImpl<SCCOp>& initialEquations,
            llvm::SmallVectorImpl<SCCOp>& equations);

        /// Collect all the initial and non-initial algorithms of a model.
        void collectAlgorithms(
            llvm::SmallVectorImpl<AlgorithmOp>& initialAlgorithms,
            llvm::SmallVectorImpl<AlgorithmOp>& algorithms);
    }];
}

def Modelica_RecordOp : Modelica_Op<"record",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica record.";

    let description = [{
        The operation represents a Modelica record.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let extraClassDeclaration = [{
        auto getVariables()
        {
            return getOps<VariableOp>();
        }
    }];
}

def Modelica_RecordCreateOp : Modelica_Op<"record_create", []>
{
    let summary = "Create record instance.";

    let description = [{
        Create an instance of a record.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $values attr-dict `:` type($values) `->` type($result)
    }];
}

def Modelica_OperatorRecordOp : Modelica_Op<"operator_record",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica operator record.";

    let description = [{
        The operation represents a Modelica operator record.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    // Ensure that the block inside the region exists.
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        Block* bodyBlock();

        auto getVariables()
        {
            return getOps<VariableOp>();
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

    let arguments = (ins SymbolNameAttr:$variable, BoolAttr:$fixed, BoolAttr:$each);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);
    }];
}

def Modelica_DefaultOp : Modelica_Op<"default",
    [ParentOneOf<["FunctionOp"]>,
    SingleBlockImplicitTerminator<"YieldOp">]>
{
    let summary = "Default attribute of a variable.";

    let description = [{
        The operation describes how to compute the default value of a variable.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);
    }];
}

def Modelica_DerOp : Modelica_Op<"der", []>
{
    let summary = "Derivative.";

    let description = [{
        The derivative operation is used to obtain the derivative of a variable.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];
}

def Modelica_TimeOp : Modelica_Op<"time", []>
{
    let summary = "Time.";

    let description = [{
        The operation represents the time of the simulation.
    }];

    let arguments = (ins);
    let results = (outs ModelicaReal);

    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let builders = [
        OpBuilder<(ins), [{
            auto realType = RealType::get($_builder.getContext());
            build($_builder, $_state, realType);
        }]>
    ];
}

def Modelica_BindingEquationOp : Modelica_Op<"binding_equation",
    [HasParent<"ModelOp">]>
{
    let summary = "Binding equation.";

    let description = [{
        The operation represents an implicit equation whose expression is
        written in form of assignment during the declaration of a variable.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);
    }];
}

def Modelica_ForEquationOp : Modelica_Op<"for_equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    NoTerminator]>
{
    let summary = "Loop defining more equation through a for loop.";

    let description = [{
        The for equation loop is an operation used to declare multiple equations whose
        structure just differ by the indices they depend on. In this sense, the loop is
        not the standard algorithmic loop developers are used to, but rather a modeling
        construct that changes the model structure.
        The iteration starts from the `from` value (included) and ends with the `to`
        value (excluded), with a increment of 'step'.
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
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "long":$from, "long":$to, "long":$step)>
    ];

    let extraClassDeclaration = [{
        Block* bodyBlock();
        Value induction();
    }];
}

def Modelica_EquationTemplateOp : Modelica_Op<"equation_template",
    [IsolatedFromAbove,
    ParentOneOf<["ModelOp"]>,
    SingleBlock,
    NoMemoryEffect]>
{
    let summary = "Equation template.";

    let description = [{
        A generic template for an equation or an initial equation.
    }];

    let arguments = (ins);
    let results = (outs EquationType:$result);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins), [{
            auto equationType = EquationType::get($_builder.getContext());
            build($_builder, $_state, equationType);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            // Create the body block of the operation.
            mlir::Block* createBody(unsigned int numOfInductions);

            /// Get the induction variables.
            mlir::ValueRange getInductionVariables();

            /// Get the implicit number of induction variables that is given by
            /// types of the variables involved in the equality.
            uint64_t getNumOfImplicitInductionVariables(
                uint64_t viewElementIndex);

            /// Get the indices implicitly required to iterate over the results
            /// of each equation side.
            /// No iteration space is returned if the equality involves scalar
            /// variables.
            std::optional<IndexSet> computeImplicitIterationSpace(
                uint64_t viewElementIndex);

            /// Get the accesses to variables performed by the equation.
            mlir::LogicalResult getAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                mlir::SymbolTableCollection& symbolTable,
                uint64_t elementIndex);

            /// Get the write accesses.
            mlir::LogicalResult getWriteAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                const IndexSet& equationIndices,
                llvm::ArrayRef<VariableAccess> accesses,
                const VariableAccess& matchedAccess);

            /// Get the read accesses.
            mlir::LogicalResult getReadAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                const IndexSet& equationIndices,
                llvm::ArrayRef<VariableAccess> accesses,
                const VariableAccess& matchedAccess);

            /// Get the IR value at a given path.
            mlir::Value getValueAtPath(const EquationPath& path);

            /// Get the access at a given path.
            std::optional<VariableAccess> getAccessAtPath(
                mlir::SymbolTableCollection& symbolTable,
                const EquationPath& path);

            /// Get the access function given the values (expressions) of the
            /// indices.
            std::optional<mlir::AffineMap> getAccessFunction(
                llvm::ArrayRef<mlir::Value> indices);

            IndexSet applyAccessFunction(
                const AccessFunction& accessFunction,
                std::optional<MultidimensionalRange> explicitEquationIndices,
                std::optional<MultidimensionalRange> implicitEquationIndices,
                const EquationPath& path);

            mlir::LogicalResult cloneWithReplacedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results);

            std::unique_ptr<AccessFunction> getReplacementTransformationAccess(
                const AccessFunction& destinationAccess,
                const AccessFunction& sourceAccess);

            mlir::LogicalResult explicitate(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> explicitEquationIndices,
                std::optional<MultidimensionalRange> implicitEquationIndices,
                const EquationPath& path);

            EquationTemplateOp cloneAndExplicitate(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> explicitEquationIndices,
                std::optional<MultidimensionalRange> implicitEquationIndices,
                const EquationPath& path);

        private:
            llvm::DenseMap<mlir::Value, unsigned int> getInductionsPositionMap();

            mlir::LogicalResult searchAccesses(
                llvm::SmallVectorImpl<VariableAccess>& accesses,
                mlir::SymbolTableCollection& symbolTableCollection,
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Value value,
                EquationPath path);

            mlir::LogicalResult searchAccesses(
                llvm::SmallVectorImpl<VariableAccess>& accesses,
                mlir::SymbolTableCollection& symbolTable,
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Value value,
                llvm::SmallVectorImpl<mlir::AffineExpr>& dimensionAccesses,
                EquationPath path);

            mlir::LogicalResult searchAccesses(
                llvm::SmallVectorImpl<VariableAccess>& accesses,
                mlir::SymbolTableCollection& symbolTable,
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Operation* op,
                llvm::SmallVectorImpl<mlir::AffineExpr>& dimensionAccesses,
                EquationPath path);

            std::optional<mlir::AffineExpr> getAffineExpr(
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Value index);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                const AccessFunction& transformation,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<std::reference_wrapper<
                    const MultidimensionalRange>> indices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                const AccessFunction& transformation,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult mapInductionVariables(
                mlir::OpBuilder& builder,
                mlir::Location loc,
                mlir::IRMapping& mapping,
                EquationTemplateOp source,
                EquationTemplateOp destination,
                const AccessFunction& transformation);

            mlir::LogicalResult explicitateLeaf(
                mlir::RewriterBase& rewriter,
                uint64_t viewElementIndex,
                size_t argumentIndex,
                EquationPath::EquationSide side);

            mlir::LogicalResult groupLeftHandSide(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> explicitEquationIndices,
                std::optional<MultidimensionalRange> implicitEquationIndices,
                const VariableAccess& requestedAccess);

            bool checkAccessEquivalence(
                const IndexSet& equationIndices,
                const VariableAccess& firstAccess,
                const VariableAccess& secondAccess);

            mlir::LogicalResult collectSummedValues(
                llvm::SmallVectorImpl<std::pair<mlir::Value, EquationPath>>& result,
                mlir::Value root,
                EquationPath path);

            std::optional<std::pair<unsigned int, mlir::Value>>
            getMultiplyingFactor(
                mlir::OpBuilder& builder,
                mlir::SymbolTableCollection& symbolTableCollection,
                llvm::DenseMap<
                    mlir::Value, unsigned int>& inductionsPositionMap,
                const IndexSet& equationIndices,
                mlir::Value value,
                llvm::StringRef variable,
                const IndexSet& variableIndices,
                EquationPath path);

        public:
    }];
}

def Modelica_EquationInstanceOp : Modelica_Op<"equation_instance",
    [ParentOneOf<["ModelOp"]>]>
{
    let arguments = (ins
        ModelicaEquation:$base,
        DefaultValuedAttr<BoolAttr, "false">:$initial,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        OptionalAttr<MultidimensionalRangeAttr>:$implicit_indices,
        OptionalAttr<I64Attr>:$view_element_index);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate,
                       "bool":$initial)>,
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        /// @name Forwarded methods.
        /// {

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        /// Get the indices visited by the induction variables.
        /// An empty iteration space is returned if there is no iteration
        /// variable.
        IndexSet getExplicitIterationSpace();

        /// Get the number of implicit induction variables.
        uint64_t getNumOfImplicitInductionVariables();

        /// Get the indices implicitly required to iterate over the results
        /// of each equation side.
        /// An empty iteration space is returned if the equality involves
        /// scalar variables.
        IndexSet getImplicitIterationSpace();

        IndexSet getIterationSpace();

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<EquationInstanceOp>& results);
    }];
}

def Modelica_MatchedEquationInstanceOp : Modelica_Op<"matched_equation_instance",
    [ParentOneOf<["ModelOp"]>]>
{
    let arguments = (ins
        ModelicaEquation:$base,
        DefaultValuedAttr<BoolAttr, "false">:$initial,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        OptionalAttr<MultidimensionalRangeAttr>:$implicit_indices,
        EquationPathAttr:$path);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate,
                       "bool":$initial,
                       "EquationPathAttr":$path)>
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        uint64_t getViewElementIndex();

        /// @name Forwarded methods.
        /// {

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        /// Get the indices visited by the induction variables.
        /// An empty iteration space is returned if there is no iteration
        /// variable.
        IndexSet getExplicitIterationSpace();

        /// Get the number of implicit induction variables.
        uint64_t getNumOfImplicitInductionVariables();

        /// Get the indices implicitly required to iterate over the results
        /// of each equation side.
        /// An empty iteration space is returned if the equality involves
        /// scalar variables.
        IndexSet getImplicitIterationSpace();

        IndexSet getIterationSpace();

        std::optional<VariableAccess> getMatchedAccess(
            mlir::SymbolTableCollection& symbolTableCollection);

       mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult explicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        MatchedEquationInstanceOp cloneAndExplicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<MatchedEquationInstanceOp>& results);
    }];
}

def Modelica_SCCOp : Modelica_Op<"scc",
    [ParentOneOf<["ModelOp"]>,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    NoTerminator]>
{
    let arguments = (ins
        DefaultValuedAttr<BoolAttr, "false">:$initial,
        DefaultValuedAttr<BoolAttr, "false">:$cycle);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        void collectEquations(
            llvm::SmallVectorImpl<ScheduledEquationInstanceOp>& equations);
    }];
}

def Modelica_ScheduledEquationInstanceOp : Modelica_Op<"scheduled_equation_instance",
    [ParentOneOf<["SCCOp"]>]>
{
    let arguments = (ins
        ModelicaEquation:$base,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        OptionalAttr<MultidimensionalRangeAttr>:$implicit_indices,
        EquationPathAttr:$path,
        TypedArrayAttrBase<
            EquationScheduleDirectionAttr,
            "Schedule iteration directions">:$iteration_directions);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate,
                       "EquationPathAttr":$path,
                       "mlir::ArrayAttr":$iterationDirections)>
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        uint64_t getViewElementIndex();

        /// @name Forwarded methods.
        /// {

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        /// Get the indices visited by the induction variables.
        /// An empty iteration space is returned if there is no iteration
        /// variable.
        IndexSet getExplicitIterationSpace();

        /// Get the number of implicit induction variables.
        uint64_t getNumOfImplicitInductionVariables();

        /// Get the indices implicitly required to iterate over the results
        /// of each equation side.
        /// An empty iteration space is returned if the equality involves
        /// scalar variables.
        IndexSet getImplicitIterationSpace();

        IndexSet getIterationSpace();

        std::optional<VariableAccess> getMatchedAccess(
            mlir::SymbolTableCollection& symbolTableCollection);

       mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult explicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        ScheduledEquationInstanceOp cloneAndExplicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<ScheduledEquationInstanceOp>& results);
    }];
}

def Modelica_EquationOp : Modelica_Op<"equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    SingleBlock]>
{
    let summary = "Equation.";

    let description = [{
        The operation represents a Modelica equation.
    }];

    let arguments = (ins DefaultValuedAttr<BoolAttr, "false">:$initial);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
       attr-dict-with-keyword $bodyRegion
    }];
}

def Modelica_EquationSideOp : Modelica_Op<"equation_side",
    [ParentOneOf<["EquationTemplateOp", "EquationOp"]>]>
{
    let summary = "Equation side.";

    let description = [{
        The operation represents one of the sides (left or right)
        of an equation.
    }];

    let arguments = (ins NonEmptyVariadic<AnyType>:$values);
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
    [ParentOneOf<["EquationTemplateOp", "EquationOp"]>,
    Terminator]>
{
    let summary = "Equation terminator.";

    let description = [{
        The operation represents the equality between two equation sides of an equation.
        It is intended to be used as terminator for the body of an equation.

        Example:

        ```mlir
        modelica.equation {
            %0 = modelica.equation_side ... : tuple<!modelica.real>
            %1 = modelica.equation_side ... : tuple<!modelica.real>
            modelica.equation_sides %0, %1 : tuple<!modelica.real>, tuple<!modelica.real>
        }
        ```
    }];

    let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
    let results = (outs);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
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

def Modelica_AlgorithmOp : Modelica_Op<"algorithm",
    [IsolatedFromAbove,
    ParentOneOf<["ModelOp", "FunctionOp"]>,
    SingleBlock,
    NoTerminator]>
{
    let summary = "Algorithm.";

    let description = [{
        The operation represents a Modelica algorithm.
    }];

    let arguments = (ins DefaultValuedAttr<BoolAttr, "false">:$initial);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
       attr-dict-with-keyword $bodyRegion
    }];
}

def Modelica_AssignmentOp : Modelica_Op<"assignment",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Assignment.";

    let description = [{
        The operation represents an assignment to be performed into an algorithmic
    }];

    let arguments = (ins AnyType:$destination, AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $destination `,` $value attr-dict `:` type($destination) `,` type($value)
    }];
}

def Modelica_FunctionOp : Modelica_Op<"function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    SymbolTable,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica function.";

    let description = [{
        The operation represents a Modelica function.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let regions = (region AnyRegion:$body);

    // Ensure that the block inside the region exists.
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        mlir::Block* bodyBlock();

        auto getVariables()
        {
            return getOps<VariableOp>();
        }

        auto getDefaultValues()
        {
            return getOps<DefaultOp>();
        }

        bool shouldBeInlined();

        /// Returns the argument types of this function.
        llvm::SmallVector<Type> getArgumentTypes();

        /// Returns the result types of this function.
        llvm::SmallVector<Type> getResultTypes();

        mlir::FunctionType getFunctionType();

        /// @name SymbolOpInterface Methods
        /// {

        bool isDeclaration() {
            return false;
        }

        /// }
    }];
}

def Modelica_DerFunctionOp : Modelica_Op<"der_function",
    [IsolatedFromAbove, Symbol]>
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
        /// @name SymbolOpInterface Methods
        /// {

        bool isDeclaration() {
            return false;
        }

        /// }
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
        A raw Modelica function consists in a Modelica function that has been converted
        to explicit control flow. It is intendend to be used only internally during the
        lowering process.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name, "mlir::FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

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
        RawFunctionOp clone(mlir::IRMapping &mapper);
        RawFunctionOp clone();

        /// Clone the internal blocks and attributes from this function into dest.
        /// Any cloned blocks are appended to the back of dest. This function
        /// asserts that the attributes of the current function and dest are
        /// compatible.
        void cloneInto(RawFunctionOp dest, mlir::IRMapping &mapper);

        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration() {
            return isExternal();
        }

        /// }
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
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def Modelica_RawVariableOp : Modelica_Op<"raw_variable", []>
{
    let summary = "Declare a variable of a class.";

    let description = [{
        TODO
    }];

    let arguments = (ins
        StrAttr:$name,
        DefaultValuedAttr<StrArrayAttr, "{}">:$dimensionsConstraints,
        Variadic<Index>:$dynamicSizes);

    let results = (outs AnyVariable:$variable);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name, "VariableType":$variableType, "::mlir::ArrayAttr":$dimensionsConstraints, "::mlir::ValueRange":$dynamicSizes), [{
            auto nameAttr = $_builder.getStringAttr(name);
            build($_builder, $_state, variableType, nameAttr, dimensionsConstraints, dynamicSizes);
    }]>];

    let extraClassDeclaration = [{
        static constexpr llvm::StringLiteral kDimensionConstraintUnbounded = "unbounded";
        static constexpr llvm::StringLiteral kDimensionConstraintFixed = "fixed";

        VariableType getVariableType() {
            return getVariable().getType().cast<VariableType>();
        }

        bool isInput() {
            return getVariableType().isInput();
        }

        bool isOutput() {
            return getVariableType().isOutput();
        }

        bool isReadOnly() {
            return getVariableType().isReadOnly();
        }

        bool isDiscrete() {
            return getVariableType().isDiscrete();
        }

        bool isParameter() {
            return getVariableType().isParameter();
        }

        bool isConstant() {
            return getVariableType().isConstant();
        }
    }];
}

def Modelica_RawVariableGetOp : Modelica_Op<"raw_variable_get", []>
{
    let summary = "Load the data held by a raw variable.";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyVariable:$variable);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$variable), [{
            auto resultType = variable.getType().cast<VariableType>().unwrap();
            build($_builder, $_state, resultType, variable);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        VariableType getVariableType() {
            return getVariable().getType().cast<VariableType>();
        }
    }];
}

def Modelica_RawVariableSetOp : Modelica_Op<"raw_variable_set", []>
{
    let summary = "Store data into a raw variable.";

    let description = [{
        Store data into the variable of a class.
    }];

    let arguments = (ins AnyVariable:$variable, AnyType:$value);
    let results = (outs);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        VariableType getVariableType() {
            return getVariable().getType().cast<VariableType>();
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

    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        OptionalAttr<FlatSymbolRefArrayAttr>:$argNames);

    let results = (outs Variadic<AnyType>);

    let assemblyFormat = [{
        $callee `(` $args `)` ($argNames^)? attr-dict `:` functional-type($args, results)
    }];

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "FunctionOp":$callee,
            CArg<"mlir::ValueRange", "{}">:$args,
            CArg<"std::optional<mlir::ArrayAttr>", "std::nullopt">:$argNames)>,
        OpBuilder<(ins
            "RawFunctionOp":$callee,
            CArg<"mlir::ValueRange", "{}">:$args)>,
        OpBuilder<(ins
            "mlir::SymbolRefAttr":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$args,
            CArg<"std::optional<mlir::ArrayAttr>", "std::nullopt">:$argNames)>
    ];

    let extraClassDeclaration = [{
        /// @name CallOpInterface
        /// {

        mlir::FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands()
        {
            return { arg_operand_begin(), arg_operand_end() };
        }

        MutableOperandRange getArgOperandsMutable()
        {
            return getArgsMutable();
        }

        operand_iterator arg_operand_begin()
        {
            return operand_begin();
        }

        operand_iterator arg_operand_end()
        {
            return operand_end();
        }

        /// Return the callee of this operation.
        mlir::CallInterfaceCallable getCallableForCallee()
        {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee)
        {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

        /// }

        mlir::Operation* getFunction(
            mlir::ModuleOp moduleOp, mlir::SymbolTableCollection& symbolTable);
    }];
}

//===---------------------------------------------------------------------===//
// Control flow operations
//===---------------------------------------------------------------------===//

def Modelica_BreakOp : Modelica_Op<"break", [Terminator]>
{
    let summary = "Loop breaking operation.";

    let description = [{
        The `modelica.break` operation terminates the execution of the closest loop that
        contains the instruction.

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

def Modelica_ReturnOp : Modelica_Op<"return", [Terminator]>
{
    let summary = "Early termination instruction for functions.";

    let description = [{
        The `modelica.return` operation terminates the execution of the whole
        function, making it returning the values as computed so far.

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

def Modelica_IfOp : Modelica_Op<"if",
    [NoTerminator,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Algorithmic if-else construct.";

    let description = [{
        The operation takes a value representing the condition and executes the code in
        the "then" region; if not, it executes the code in the optional "else" region.
        The value used as condition must be convertible to `!modelica.bool`.

        Example:

        ```mlir
        %0 = ... : !modelica.bool

        if (%0 : !modelica.bool) {
            ...
        } else {
            ...
        }
        ```
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

def Modelica_ForOp : Modelica_Op<"for",
    [LoopLikeOpInterface,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "For algorithmic loop.";

    let description = [{
        The operation represents an algorithmic for loop.
        It consists in three regions:
            1. A condition region that determines whether the iterations should go on.
            2. The body region to be executed at each iteration.
            3. A post-iteration region that is executed after each execution of the body
               region.

        Example:

        ```mlir
        modelica.for condition {
            %0 = ... : !modelica.bool
            modelica.condition (%0 : !modelica.bool)
        } body {
            modelica.yield
        } step {
            modelica.yield
        }
        ```
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
        /// @name LoopLikeOpInterface methods.
        /// {

        mlir::Region& getLoopBody()
        {
            return getBodyRegion();
        }

        /// }

        Block* conditionBlock();
        Block* bodyBlock();
        Block* stepBlock();
    }];
}

def Modelica_WhileOp : Modelica_Op<"while",
    [NoTerminator,
    LoopLikeOpInterface,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "While algorithmic loop.";

    let description = [{
        The operation represents an algorithmic while loop.
        It consists in two regions:
            1. A condition region that determines whether the iterations should go on.
            2. The body region to be executed at each iteration.

        Example:

        ```mlir
        modelica.while {
            %0 = ... : !modelica.bool
            modelica.condition (%0 : !modelica.bool)
        } do {
            ...
        }
    }
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        /// @name LoopLikeOpInterface methods.
        /// {

        mlir::Region& getLoopBody()
        {
            return getBodyRegion();
        }

        /// }
    }];
}

def Modelica_ConditionOp : Modelica_Op<"condition",
    [Terminator,
    ParentOneOf<["ForOp", "WhileOp"]>]>
{
    let summary = "Iteration condition.";

    let description = [{
        The operation represents the condition to be evaluated at each iteration of the
        parent loop (which can be either a `modelica.for` or `modelica.while`).


    }];

    let arguments = (ins ModelicaBoolean:$condition, Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        `(` $condition `:` type($condition) `)` attr-dict ($values^ `:` type($values))?
    }];

    let builders = [
        OpBuilder<(ins "::mlir::Value":$condition), [{
            build($_builder, $_state, condition, std::nullopt);
        }]>
    ];
}

//===---------------------------------------------------------------------===//
// Utility operations
//===---------------------------------------------------------------------===//

def Modelica_YieldOp : Modelica_Op<"yield",
    [Terminator,
    ParentOneOf<["ForEquationOp", "IfOp", "ForOp", "WhileOp", "ModelOp",
                 "VariableOp", "StartOp", "DefaultOp", "BindingEquationOp",
                 "ReductionOp"]>]>
{
    let summary = "Yield values to the parent operation.";

    let description = [{
        The operation yields zero or more SSA values from an op region and
        terminates the region. The semantics of how the yielded values are used
        is defined by the parent operation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        attr-dict ($values^ `:` type($values))?
    }];

    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Modelica_CastOp : Modelica_Op<"cast",
    [NoMemoryEffect,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

    let arguments = (ins AnyModelicaCompatibleScalar:$value);
    let results = (outs AnyModelicaCompatibleScalar:$result);

    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
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
        $value attr-dict `:` type($value)
    }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
