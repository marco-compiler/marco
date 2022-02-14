#ifndef MARCO_DIALECTS_MODELICA_MODELICA_TD
#define MARCO_DIALECTS_MODELICA_MODELICA_TD

include "marco/dialects/modelica/ModelicaAttributes.td"
include "marco/dialects/modelica/ModelicaDialect.td"
include "marco/dialects/modelica/ModelicaInterfaces.td"
include "marco/dialects/modelica/ModelicaTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

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

// TODO ModelOp
// TODO ForEquationOp
// TODO EquationOp
// TODO EquationSidesOp
// TODO FunctionOp
// TODO FunctionTerminatorOp
// TODO DerFunctionOp
// TODO ConstantOp
// TODO CastOp
// TODO AssignmentOp
// TODO CallOp
// TODO MemberCreateOp
// TODO MemberLoadOp
// TODO MemberStoreOp
// TODO AllocaOp
// TODO AllocOp
// TODO MinOp
// TODO MaxOp
// TODO DerSeedOp

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

def Modelica_ArrayCastOp : Modelica_Op<"array_cast",
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ArrayCloneOp : Modelica_Op<"array_clone",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<HeapAllocator>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];

    let extraClassDeclaration = [{
        bool canSourceBeForwarded() const;
    }];
}

def Modelica_DimOp : Modelica_Op<"dim", []>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs);

    let assemblyFormat = [{
        $array attr-dict `:` type($array)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_LoadOp : Modelica_Op<"load",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array, Variadic<Index>:$indexes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $indexes `]` attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_StoreOp : Modelica_Op<"store",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array, Variadic<Index>:$indexes, AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $array `[` $indexes `]` `,` $value attr-dict `:` type($array) `,` type($value)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_SubscriptionOp : Modelica_Op<"subscription",
    [DeclareOpInterfaceMethods<ViewLikeOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array, Variadic<Index>:$indexes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `[` $indexes `]` attr-dict `:` type($array) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

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
        The `modelica.add` operation computes the sum of two values.
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
        The `modelica.eq` operation checks whether two scalar values are equal.

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
        The `modelica.neq` operation checks whether two scalar values are not equal.

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
        The `modelica.gt` operation checks whether a scalar value is greater
        than another one.

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
        The `modelica.gte` operation checks whether a scalar value is greater
        or equal to another one.

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
        The `modelica.lt` operation checks whether a scalar value is less
        than another one.

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
        The `modelica.lte` operation checks whether a scalar value is less
        or equal to another one.

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
        The `modelica.not` operation computes the logical NOT of a boolean value.
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
        The `modelica.and` operation computes the logical AND between two boolean values.
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
        The `modelica.or` operation computes the logical OR between two boolean values.
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

def Modelica_AcosOp : Modelica_Op<"acos",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_AsinOp : Modelica_Op<"asin",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_AtanOp : Modelica_Op<"atan",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

// TODO derivable

def Modelica_Atan2Op : Modelica_Op<"atan2",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
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

def Modelica_CoshOp : Modelica_Op<"cosh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_DiagonalOp : Modelica_Op<"diagonal",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $exponent attr-dict `:` type($exponent) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_FillOp : Modelica_Op<"fill", []>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array, AnyType:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `,` $value attr-dict `:` `(` type($array) `,` type($value) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_IdentityOp : Modelica_Op<"identity",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$begin, AnyType:$end, AnyType:$steps);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $begin `,` $end `,` $steps attr-dict `:` `(` type($begin) `,` type($end) `,` type($steps) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_LogOp : Modelica_Op<"log",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_Log10Op : Modelica_Op<"log10",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_NDimsOp : Modelica_Op<"ndims", []>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "Returns an array with '1' elements";

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
    let summary = "TODO";

    let description = [{
        TODO
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

def Modelica_SinOp : Modelica_Op<"sin",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_SinhOp : Modelica_Op<"sinh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_SizeOp : Modelica_Op<"size",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$array, Optional<AnyType>:$dimension);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array `,` $dimension attr-dict `:` type($array) `,` type($dimension) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

// TODO DeclareOpInterfaceMethods<DerivableOpInterface>
def Modelica_SqrtOp : Modelica_Op<"sqrt",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>]>
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

def Modelica_SumOp : Modelica_Op<"sum", []>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "TODO";

    let description = [{
        TODO
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

def Modelica_TanhOp : Modelica_Op<"tanh",
    [DeclareOpInterfaceMethods<VectorizableOpInterface>,
    DeclareOpInterfaceMethods<DerivableOpInterface>]>
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

def Modelica_TransposeOp : Modelica_Op<"transpose", []>
{
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "Returns an array with '0' elements";

    let description = [{
        Returns an array with "0" elements
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

//===----------------------------------------------------------------------===//
// Control flow operations
//===----------------------------------------------------------------------===//

def Modelica_BreakOp : Modelica_Op<"break", [Terminator]>
{
    let summary = "TODO";

    let description = [{
        TODO
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

    let arguments = (ins AnyType:$condition);
    let results = (outs);

    let assemblyFormat = [{
        $condition attr-dict `:` type($condition)
    }];

    let verifier = [{ return ::verify(*this); }];
}

def Modelica_ForOp : Modelica_Op<"for",
    [DeclareOpInterfaceMethods<DerivableOpInterface>]>
{
    let summary = "TODO";

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
    let summary = "TODO";

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
    let summary = "TODO";

    let description = [{
        TODO
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
    let summary = "TODO";

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

def Modelica_PrintOp : Modelica_Op<"print", []>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs);

    let assemblyFormat = [{
        $value attr-dict `:` type($value)
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
