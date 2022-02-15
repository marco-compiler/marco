#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "marco/dialects/modelica/ModelicaAttributes.td"
include "marco/dialects/modelica/ModelicaDialect.td"
include "marco/dialects/modelica/ModelicaInterfaces.td"
include "marco/dialects/modelica/ModelicaTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"

// Modelica Boolean type
def Boolean : Type<CPred<"$_self.isa<::mlir::modelica::BooleanType>()">,
                   "boolean", "::mlir::modelica::BooleanType">;

// Modelica Integer type
def Integer : Type<CPred<"$_self.isa<::mlir::modelica::IntegerType>()">,
                   "integer", "::mlir::modelica::IntegerType">;

// Modelica Real type
def Real : Type<CPred<"$_self.isa<::mlir::modelica::RealType>()">,
                "real", "::mlir::modelica::RealType">;

// Modelica-compatible scalar type
def AnyScalar : AnyTypeOf<[Boolean, Integer, Real, Index]>;

// Whether a type is an ArrayType.
def IsArrayTypePred : CPred<"$_self.isa<::mlir::modelica::ArrayType>()">;

class ArrayOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsArrayTypePred,
                  "$_self.cast<::mlir::modelica::ArrayType>().getElementType()",
                  "array", "::mlir::modelica::ArrayType">;

def AnyArray : ArrayOf<[Boolean, Integer, Real, Index]>;

//===----------------------------------------------------------------------===//
// Modelica operation definitions
//===----------------------------------------------------------------------===//

class Modelica_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Modelica_Dialect, mnemonic, traits>;

// Base class for unary operations.
class Modelica_UnaryOp<string mnemonic, list<OpTrait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for binary operations.
class Modelica_BinaryOp<string mnemonic, list<OpTrait> traits = []> :
    Modelica_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// TODO FunctionOp
// TODO FunctionTerminatorOp
// TODO DerFunctionOp
// TODO AssignmentOp
// TODO CallOp
// TODO DerSeedOp

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
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` type($array)
    }];

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
    }]>];

    let extraClassDeclaration = [{
        ArrayType getArrayType() { return array().getType().cast<ArrayType>(); }
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_AllocOp : Modelica_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Allocate an array on the heap.";

    let description = [{
        Allocate an array on the heap.
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` type($array)
    }];

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
    }]>];

    let extraClassDeclaration = [{
        ArrayType getArrayType() { return array().getType().cast<ArrayType>(); }
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ArrayCastOp : Modelica_Op<"array_cast",
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>]>
{
    let summary = "Cast operation for array types.";

    let description = [{
        This operation should be used only for two purposes: the first one
        is for function calls, to remove the allocation scope before passing
        the array as arguments to the functions, or to generalize the sizes
        to unknown ones; the second one is to cast from an array with unknown
        allocation scope to an array with a known one.
        The operation is NOT intended to be used to change the allocation scope
        between known ones (i.e. stack -> heap or heap -> stack), to cast the
        element type to a different one or to to specialize the shape to a
        fixed one.

        Correct usage examples:

        ```mlir
        %0 = ... : !modelica.array<heap, 3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<heap, 3x2x!modelica.int> -> !modelica.array<3x2x!modelica.int>
        %2 = modelica.array_cast %0 : !modelica.array<heap, 3x2x!modelica.int> -> !modelica.array<heap, ?x?x!modelica.int>
        %3 = modelica.array_cast %0 : !modelica.array<heap, 3x2x!modelica.int> -> !modelica.array<*x!modelica.int>
        ```

        Wrong usage examples:

        ```mlir
        %0 = ... : !modelica.array<heap, 3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<heap, 3x2x!modelica.int> -> !modelica.array<heap, 3x2x!modelica.real>
        %2 = modelica.array_cast %0 : !modelica.array<heap, 3x2x!modelica.int> -> !modelica.array<stack, 3x2x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs AnyArray:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];

    let extraClassDeclaration = [{
        ArrayType getSourceArrayType() {
            return array().getType().cast<ArrayType>();
        }

        ArrayType getResultArrayType() {
            return result().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayCloneOp : Modelica_Op<"array_clone",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<HeapAllocator>]>
{
    let summary = "Create a clone of an array.";

    let description = [{
        Create an array with the same dimensions of the source one
        and copy its values. The allocation scope may be different
        between them.

        Example:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.real>
        %1 = modelica.array_clone %0 : !modelica.array<stack, 3x!modelica.real> -> !modelica.array<heap, 3x!modelica.real>
        ```
    }];

    let arguments = (ins AnyArray:$source);
    let results = (outs AnyArray:$result);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];

    let extraClassDeclaration = [{
        bool canSourceBeForwarded() const;
    }];
}

def Modelica_ArrayFillOp : Modelica_Op<"array_fill",
    [TypesMatchWith<"type of 'value' matches element type of 'array'",
                    "array", "value",
                    "$_self.cast<ArrayType>().getElementType()">]>
{
    let summary = "Fill an array with a value";

    let description = [{
        Set all the elements of an array to a given value.
        The value to be set must have the same type of array elemments.

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
        $array `,` $value attr-dict `:` type($array)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_DimOp : Modelica_Op<"dim", []>
{
    let summary = "Get the size of an array dimension.";

    let description = [{
        Get the size of an array dimension. The operation is trivial in case of
        statically shaped arrays, while it is more useful in case of dimensions
        that are unknown at compile-time.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x?x?x!modelica.int>
        %c2 = constant 2 : index
        %dim = modelica.dim %array, $c2 : !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyType:$array, Index:$dimension);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `,` $dimension attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_FreeOp : Modelica_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocate an heap-allocated array.";

    let description = [{
        Deallocate an heap-allocated array.

        Example:

        ```mlir
        %array = ... : !modelica.array<heap, ?x!modelica.int>
        modelica.free %array : !modelica.array<heap, ?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs);

    let assemblyFormat = [{
        $array attr-dict `:` type($array)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        arguments provided within brackets needs to match the rank of the array.

        Example:

        ```mlir
        $array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = modelica.load $array[%0, %1] : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyArray:$array, Variadic<Index>:$indexes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $indexes `]` attr-dict `:` type($array)
    }];

    let verifier = [{ return ::verify(*this); }];

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
            return array().getType().cast<ArrayType>();
        }

        operand_range getIndices() {
            return { operand_begin() + 1, operand_end() };
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
        Store a value to an array location given by indices. The value stored should
        have the same type as the elemental type of the array. The number of
        arguments provided within brackets needs to match the rank of the array.

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
        Variadic<Index>:$indexes);

    let results = (outs);

    let assemblyFormat = [{
        $array `[` $indexes `]` `,` $value attr-dict `:` type($array)
    }];

    let verifier = [{ return ::verify(*this); }];

    let extraClassDeclaration = [{
        ArrayType getArrayType() {
            return array().getType().cast<ArrayType>();
        }

        operand_range getIndices() {
            return { operand_begin() + 2, operand_end() };
        }
      }];
}

def Modelica_SubscriptionOp : Modelica_Op<"subscription",
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>,
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

    let arguments = (ins AnyType:$array, Variadic<Index>:$indices);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $indices `]` attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Member operations
//===----------------------------------------------------------------------===//

def Modelica_MemberCreateOp : Modelica_Op<"member_create",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Declare a member of a class.";

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

    let arguments = (ins StrAttr:$name, Variadic<Index>:$dynamicSizes, BoolAttr:$isConstant);
    let results = (outs AnyArray:$member);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` type($member)
    }];

    let builders = [
        OpBuilder<(ins "MemberType":$memberType, "ValueRange":$dynamicSizes, "BoolAttr":$isConstant), [{
            build($_builder, $_state, memberType, dynamicSizes, isConstant);
    }]>];

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return member().getType().cast<MemberType>();
        }
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_MemberLoadOp : Modelica_Op<"member_load",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Load the data held by a class member.";

    let description = [{
        Load the data held by member of a class.
    }];

    let arguments = (ins AnyType:$member);
    let results = (outs AnyType);

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return member().getType().cast<MemberType>();
        }
      }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_MemberStoreOp : Modelica_Op<"member_store",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Store data into a class member.";

    let description = [{
        Store data into the member of a class.
    }];

    let arguments = (ins AnyType:$member, AnyType:$value);
    let results = (outs);

    let extraClassDeclaration = [{
        MemberType getMemberType() {
            return member().getType().cast<MemberType>();
        }
      }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

def Modelica_ConstantOp : Modelica_Op<"constant",
    [ConstantLike, NoSideEffect]>
{
    let summary = "Constant.";

    let description = [{
        The operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute.

        Example:

        ```mlir
        %0 = modelica.constant #modelica.int<3> : !modelica.int
        ```
    }];

    let arguments = (ins AnyAttr:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilder<(ins "Attribute":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];

    let verifier = [{ return ::verify(*this); }];
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
    let summary = "Sum two values.";

    let description = [{
        The operation computes the sum of two values.
        It takes two operands and returns one result. The type of the operands can
        be different but yet must be shape-compatible. Also array values are allowed,
        and in this case the result must have the same shape as the operands.
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_AddEWOp : Modelica_Op<"add_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_MulEWOp : Modelica_Op<"mul_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand  attr-dict `:` type($operand) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_PowOp : Modelica_Op<"pow",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_PowEWOp : Modelica_Op<"pow_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SubOp : Modelica_Op<"sub",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SubEWOp : Modelica_Op<"sub_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InvertibleOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

// Base class for comparison operations.
class Modelica_ComparisonOp<string mnemonic, list<OpTrait> traits = []> :
    Modelica_Op<mnemonic, traits> {
    let arguments = (ins
        AnyType:$lhs,
        AnyType:$rhs);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
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
}

def Modelica_LteOp : Modelica_Op<"lte", []>
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Built-in operations
//===----------------------------------------------------------------------===//

def Modelica_AbsOp : Modelica_Op<"abs",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_AcosOp : Modelica_Op<"acos",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_AsinOp : Modelica_Op<"asin",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_AtanOp : Modelica_Op<"atan",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

// TODO Make atan2 derivable

def Modelica_Atan2Op : Modelica_Op<"atan2",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_CosOp : Modelica_Op<"cos",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_CoshOp : Modelica_Op<"cosh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
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
        $values attr-dict `:` type($values) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ExpOp : Modelica_Op<"exp",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
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
        $size attr-dict `:` type($size) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_LogOp : Modelica_Op<"log",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_Log10Op : Modelica_Op<"log10",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ArrayMaxOp : Modelica_Op<"array_max", []>
{
    let summary = "Get the greatest value inside an array.";

    let description = [{
        Get the greatest value among an array or two values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.array_max %0 : !modelica.array<5x!modelica.int> -> !modelica.int
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs AnyScalar:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_MaxOp : Modelica_Op<"max", []>
{
    let summary = "Get the greatest value among two scalars.";

    let description = [{
        Get the greatest value among two scalars.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.max %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyScalar:$first, AnyScalar:$second);
    let results = (outs AnyScalar:$result);

    let assemblyFormat = [{
        $first `,` $second attr-dict `:` `(` type($first) `,` type($second) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ArrayMinOp : Modelica_Op<"array_min", []>
{
    let summary = "Get the smallest value inside an array.";

    let description = [{
        Get the smallest value among an array or two values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.array_min %0 : !modelica.array<5x!modelica.int> -> !modelica.int
        ```
    }];

    let arguments = (ins AnyArray:$array);
    let results = (outs AnyScalar:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_MinOp : Modelica_Op<"min", []>
{
    let summary = "Get the smallest value among two scalars.";

    let description = [{
        Get the smallest value among two scalars.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.min %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyScalar:$first, AnyScalar:$second);
    let results = (outs AnyScalar:$result);

    let assemblyFormat = [{
        $first `,` $second attr-dict `:` `(` type($first) `,` type($second) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $sizes attr-dict `:` type($sizes) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SignOp : Modelica_Op<"sign",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SinOp : Modelica_Op<"sin",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SinhOp : Modelica_Op<"sinh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
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

        In the second case the second operand also specifies the dimension
        of interested, and thus the operation returns just a scalar value
        corresponding to the size of that dimension.

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.int>
        %1 = constant 1 : index
        %2 = modelica.size %0, %1 : (!modelica.array<?x?x!modelica.int>, index) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$array, Optional<AnyType>:$dimension);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `,` $dimension attr-dict `:` type($array) `,` type($dimension) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

// TODO make sqrt derivable

def Modelica_SqrtOp : Modelica_Op<"sqrt",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
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

    let verifier = [{ return ::verify(*this); }];
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
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $matrix attr-dict `:` type($matrix) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_TanOp : Modelica_Op<"tan",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_TanhOp : Modelica_Op<"tanh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
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

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_TransposeOp : Modelica_Op<"transpose", []>
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

    let verifier = [{ return ::verify(*this); }];
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
        $sizes attr-dict `:` type($sizes) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
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
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ModelOp : Modelica_Op<"model",
    [IsolatedFromAbove,
    DeclareOpInterfaceMethods<RegionKindInterface>]>
{
    let summary = "TODO";

    let description = [{
        The operation represents a Modelica model.
    }];

    let arguments = (ins
        RealAttr:$startTime,
        RealAttr:$endTime,
        RealAttr:$timeStep);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$initRegion,
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        attr-dict regions
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
        with the `to` value (excluded).
        Multiple for loops can be nested into each other.

        Example:

        ```mlir
        modelica.for_equation {from = 3, to = 9} {
            modelica.equation {
                ...
            }
        }
        ```
    }];

    let arguments = (ins
        IndexAttr:$from,
        IndexAttr:$to);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def Modelica_EquationOp : Modelica_Op<"equation",
    [ParentOneOf<["ModelOp", "ForEquationOp"]>,
    SingleBlockImplicitTerminator<"EquationSidesOp">]>
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

    let assemblyFormat = [{
        attr-dict regions
    }];
}

def Modelica_EquationSideOp : Modelica_Op<"equation_side",
    [HasParent<"EquationOp">, Terminator]>
{
    let summary = "Equation side.";

    let description = [{
        The operation represents one of the sides (left or right)
        of an equation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyTuple);
}

def Modelica_EquationSidesOp : Modelica_Op<"equation_sides",
    [HasParent<"EquationOp">, Terminator]>
{
    let summary = "Equation.";

    let description = [{
        The operation represents a Modelica equation.
    }];

    let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
    let results = (outs);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
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

    let arguments = (ins Boolean:$condition);
    let results = (outs);

    let assemblyFormat = [{
        $condition attr-dict `:` type($condition)
    }];
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
}

def Modelica_IfOp : Modelica_Op<"if",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "Algorithmic if-else construct.";

    let description = [{
        TODO
    }];

    let arguments = (ins Variadic<AnyType>:$args);
    let results = (outs);

    let regions = (region
      SizedRegion<1>:$thenRegion,
      AnyRegion:$elseRegion
    );
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
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

    let assemblyFormat = [{
        attr-dict regions
    }];
}

// Parent: "ForEquationOp", "IfOp", "ForOp", "WhileOp", "ModelOp"
def Modelica_YieldOp : Modelica_Op<"yield",
    [Terminator,
    ParentOneOf<["IfOp", "ForOp", "WhileOp"]>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = [{
        $values attr-dict `:` type($values)
    }];
}

//===----------------------------------------------------------------------===//
// Utility operations
//===----------------------------------------------------------------------===//

def Modelica_CastOp : Modelica_Op<"cast", []>
{
    let summary = "Cast a value to another type.";

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

    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_PrintOp : Modelica_Op<"print", []>
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

    let verifier = [{ return ::verify(*this); }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
