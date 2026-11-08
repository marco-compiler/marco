#ifndef MARCO_DIALECT_MODELICA_IR_OPS_TD
#define MARCO_DIALECT_MODELICA_IR_OPS_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "marco/Dialect/Modelica/IR/Attributes.td"
include "marco/Dialect/Modelica/IR/Enums.td"
include "marco/Dialect/Modelica/IR/Types.td"
include "marco/Dialect/Modelica/IR/OpInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

def Modelica_Boolean
    : Type<CPred<"::mlir::isa<::mlir::modelica::BooleanType>($_self)">,
           "Boolean", "::mlir::modelica::BooleanType">,
      BuildableType<"::mlir::modelica::BooleanType::get($_builder.getContext())">;

def Modelica_Integer
    : Type<CPred<"::mlir::isa<::mlir::modelica::IntegerType>($_self)">,
           "Integer", "::mlir::modelica::IntegerType">,
      BuildableType<"::mlir::modelica::IntegerType::get($_builder.getContext())">;

def Modelica_Real
    : Type<CPred<"::mlir::isa<::mlir::modelica::RealType>($_self)">,
           "Real", "::mlir::modelica::RealType">,
      BuildableType<"::mlir::modelica::RealType::get($_builder.getContext())">;

def Modelica_String
    : Type<CPred<"::mlir::isa<::mlir::modelica::StringType>($_self)">,
           "String", "::mlir::modelica::StringType">,
      BuildableType<"::mlir::modelica::StringType::get($_builder.getContext())">;

def Modelica_Enumeration
    : Type<CPred<"::mlir::isa<::mlir::modelica::EnumerationType>($_self)">,
           "enumeration", "::mlir::modelica::EnumerationType">;

def Modelica_Object
    : Type<CPred<"::mlir::isa<::mlir::modelica::ObjectType>($_self)">,
           "object", "::mlir::modelica::ObjectType">;

// Any scalar value.
def Modelica_Scalar : AnyTypeOf<[
        Modelica_Boolean,
        Modelica_Integer,
        Modelica_Real,
        Modelica_String,
        Modelica_Enumeration,
        Modelica_Object
    ]>;

// Any scalar value or array of scalar values.
def Modelica_Value : AnyTypeOf<[
        Modelica_Scalar,
        RankedTensorOf<[Modelica_Scalar]>
    ]>;

//===---------------------------------------------------------------------===//
// Base classes
//===---------------------------------------------------------------------===//

class Modelica_Op<string mnemonic, list<Trait> traits = []>
    : Op<Modelica_Dialect, mnemonic, traits>;

// Base classes for the operations computing a single value from the
// operands. The operations deriving from them are expected to be free of
// side effects and to infer their result type. They are not marked as
// speculatable because many of them have undefined behaviour for some
// inputs (e.g., division by zero).

class Modelica_NullaryOp<string mnemonic, list<Trait> traits = []>
    : Modelica_Op<mnemonic, traits>
{
    let arguments = (ins);
    let results = (outs Modelica_Value:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

class Modelica_UnaryOp<string mnemonic, list<Trait> traits = []>
    : Modelica_Op<mnemonic, traits>
{
    let arguments = (ins Modelica_Value:$operand);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` functional-type(operands, results)
    }];
}

class Modelica_BinaryOp<string mnemonic, list<Trait> traits = []>
    : Modelica_Op<mnemonic, traits>
{
    let arguments = (ins Modelica_Value:$lhs, Modelica_Value:$rhs);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
    }];
}

class Modelica_VariadicOp<string mnemonic, list<Trait> traits = []>
    : Modelica_Op<mnemonic, traits>
{
    let arguments = (ins Variadic<Modelica_Value>:$values);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $values attr-dict `:` functional-type(operands, results)
    }];
}

//===---------------------------------------------------------------------===//
// Constants and terminators
//===---------------------------------------------------------------------===//

def Modelica_ConstantOp : Modelica_Op<"constant", [
        ConstantLike,
        Pure,
        AllTypesMatch<["value", "result"]>
    ]>
{
    let summary = "Constant value.";

    let description = [{
        The operation turns a literal of the dialect into an SSA value.
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = "$value attr-dict";
    let hasFolder = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::TypedAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];
}

def Modelica_YieldOp : Modelica_Op<"yield", [Pure, Terminator]>
{
    let summary = "Yield values from a region.";

    let description = [{
        The operation terminates the regions of the dialect operations, and
        yields the values computed by the region to the enclosing operation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);

    let assemblyFormat = "($values^ `:` type($values))? attr-dict";

    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, ::mlir::ValueRange()); }]>
    ];
}

//===---------------------------------------------------------------------===//
// Arithmetic operators
//===---------------------------------------------------------------------===//

def Modelica_NegateOp : Modelica_UnaryOp<"neg", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Unary minus.";

    let description = [{
        Element-wise negation of a numeric scalar or array.
    }];

    let hasFolder = 1;
}

def Modelica_AddOp : Modelica_BinaryOp<"add", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Addition and String concatenation (`+`).";

    let description = [{
        Element-wise addition of numeric scalars or arrays, or element-wise
        concatenation of String scalars or arrays. The operands must have the
        same shape.
    }];

    let hasFolder = 1;
}

def Modelica_SubOp : Modelica_BinaryOp<"sub", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Subtraction (`-`).";

    let description = [{
        Element-wise subtraction of numeric scalars or arrays. The operands must
        have the same shape.
    }];

    let hasFolder = 1;
}

def Modelica_MulOp : Modelica_BinaryOp<"mul", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Multiplication (`*`).";

    let description = [{
        Multiplication of numeric values. If one of the operands is a scalar,
        the multiplication is performed element-wise. Otherwise, the operation
        is a scalar product of vectors, a vector-matrix, matrix-vector or
        matrix-matrix product.
    }];

    let hasFolder = 1;
}

def Modelica_DivOp : Modelica_BinaryOp<"div", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Division (`/`).";

    let description = [{
        Element-wise division of a numeric scalar or array by a numeric scalar.
        The result is always Real.
    }];

    let hasFolder = 1;
}

def Modelica_PowOp : Modelica_BinaryOp<"pow", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Exponentiation (`^`).";

    let description = [{
        Exponentiation of numeric scalars, whose result is always Real, or
        exponentiation of a square numeric matrix by a non-negative Integer
        scalar, whose result has the type of the matrix.
    }];

    let hasFolder = 1;
}

def Modelica_AddEWOp : Modelica_BinaryOp<"add_ew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Element-wise addition and String concatenation (`.+`).";

    let description = [{
        Element-wise addition of numeric values, or element-wise concatenation
        of String values. The operands must have the same shape, or one of them
        must be a scalar.
    }];

    let hasFolder = 1;
}

def Modelica_SubEWOp : Modelica_BinaryOp<"sub_ew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Element-wise subtraction (`.-`).";

    let description = [{
        Element-wise subtraction of numeric values. The operands must have the
        same shape, or one of them must be a scalar.
    }];

    let hasFolder = 1;
}

def Modelica_MulEWOp : Modelica_BinaryOp<"mul_ew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Element-wise multiplication (`.*`).";

    let description = [{
        Element-wise multiplication of numeric values. The operands must have
        the same shape, or one of them must be a scalar.
    }];

    let hasFolder = 1;
}

def Modelica_DivEWOp : Modelica_BinaryOp<"div_ew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Element-wise division (`./`).";

    let description = [{
        Element-wise division of numeric values. The operands must have the same
        shape, or one of them must be a scalar. The result is always Real.
    }];

    let hasFolder = 1;
}

def Modelica_PowEWOp : Modelica_BinaryOp<"pow_ew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Element-wise exponentiation (`.^`).";

    let description = [{
        Element-wise exponentiation of numeric values. The operands must have
        the same shape, or one of them must be a scalar. The result is always
        Real.
    }];

    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// Relational and logical operators
//===---------------------------------------------------------------------===//

class Modelica_RelationalOp<string mnemonic, string symbol>
    : Modelica_BinaryOp<mnemonic, [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Relational operator (`" # symbol # "`).";

    let description = [{
        Relational operator between scalars of simple types. Integer and Real
        values can be compared with each other. Boolean values are ordered as
        `false < true`, String values are ordered lexicographically and
        enumeration literals according to their order of declaration.
    }];

    let hasFolder = 1;
}

def Modelica_EqOp : Modelica_RelationalOp<"eq", "==">;
def Modelica_NotEqOp : Modelica_RelationalOp<"ne", "<>">;
def Modelica_LtOp : Modelica_RelationalOp<"lt", "<">;
def Modelica_LteOp : Modelica_RelationalOp<"le", "<=">;
def Modelica_GtOp : Modelica_RelationalOp<"gt", ">">;
def Modelica_GteOp : Modelica_RelationalOp<"ge", ">=">;

def Modelica_NotOp : Modelica_UnaryOp<"not", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Logical negation.";

    let description = [{
        Element-wise logical negation of a Boolean scalar or array.
    }];

    let hasFolder = 1;
}

def Modelica_AndOp : Modelica_BinaryOp<"and", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Logical conjunction.";

    let description = [{
        Element-wise logical conjunction of Boolean scalars or arrays having the
        same shape.
    }];

    let hasFolder = 1;
}

def Modelica_OrOp : Modelica_BinaryOp<"or", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Logical disjunction.";

    let description = [{
        Element-wise logical disjunction of Boolean scalars or arrays having the
        same shape.
    }];

    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// If-expression
//===---------------------------------------------------------------------===//

def Modelica_IfExprOp : Modelica_Op<"if_expr", [
        RecursiveMemoryEffects,
        NoRegionArguments
    ]>
{
    let summary = "If-expression.";

    let description = [{
        The operation represents an if-expression. Only the region corresponding
        to the value of the condition is evaluated, and the value it yields
        becomes the result of the operation. Expressions with `elseif` are
        represented by nesting another `if_expr` into the `else` region.

        The yielded values must be type compatible with the result: Integer
        values can be yielded when the result is Real.
    }];

    let arguments = (ins Modelica_Boolean:$condition);
    let results = (outs Modelica_Value:$result);

    let regions = (region
        SizedRegion<1>:$thenRegion,
        SizedRegion<1>:$elseRegion
    );

    let assemblyFormat = [{
        $condition `:` type($result) $thenRegion `else` $elseRegion attr-dict
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//===---------------------------------------------------------------------===//
// Array constructors
//===---------------------------------------------------------------------===//

def Modelica_RangeOp : Modelica_Op<"range", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Range (`start:stop` and `start:step:stop`).";

    let description = [{
        Vector of values going from `start` to `stop`, with an increment of
        `step`. The step is 1 if not specified. Ranges of Boolean values and of
        enumeration literals are also allowed, but without explicit step.
    }];

    let arguments = (ins
        Modelica_Scalar:$start,
        Modelica_Scalar:$stop,
        Optional<Modelica_Scalar>:$step
    );

    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $start `to` $stop (`step` $step^)? attr-dict
        `:` functional-type(operands, results)
    }];

    let hasFolder = 1;
}

def Modelica_ArrayOp : Modelica_VariadicOp<"array", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Array constructor (`{a, b, ...}`).";

    let description = [{
        Array whose elements are the operands. The operands must have the same
        shape, and the result has one more dimension than them. Integer and Real
        elements can be mixed, leading to a Real array.
    }];

    let hasFolder = 1;
}

def Modelica_CatOp : Modelica_Op<"cat", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Concatenation along a dimension (`cat`).";

    let description = [{
        Concatenation of arrays along the zero-based dimension `dim`. The arrays
        must have the same rank and the same sizes along all the other
        dimensions.
    }];

    let arguments = (ins Modelica_Integer:$dim, Variadic<Modelica_Value>:$values);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $dim `,` $values attr-dict `:` functional-type(operands, results)
    }];
}

def Modelica_PromoteOp : Modelica_Op<"promote", [NoMemoryEffect]>
{
    let summary = "Append dimensions of size 1 (`promote`).";

    let description = [{
        Appends dimensions of size 1 to the right of the operand, so that the
        result has `rank` dimensions. The operation is used to represent the
        concatenation operator `[a, b; c, d]`.
    }];

    let arguments = (ins Modelica_Value:$operand, I64Attr:$rank);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $operand `,` $rank attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

def Modelica_IdentityOp : Modelica_UnaryOp<"identity", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Identity matrix (`identity`).";

    let description = [{
        The `n x n` Integer identity matrix.
    }];
}

def Modelica_DiagonalOp : Modelica_UnaryOp<"diagonal", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Diagonal matrix (`diagonal`).";

    let description = [{
        Square matrix having the elements of the vector operand on the diagonal,
        and zero elsewhere.
    }];
}

def Modelica_ZerosOp : Modelica_VariadicOp<"zeros", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Array of zeros (`zeros`).";

    let description = [{
        Integer array of zeros whose dimensions are given by the operands.
    }];
}

def Modelica_OnesOp : Modelica_VariadicOp<"ones", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Array of ones (`ones`).";

    let description = [{
        Integer array of ones whose dimensions are given by the operands.
    }];
}

def Modelica_FillOp : Modelica_Op<"fill", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Array with all elements equal (`fill`).";

    let description = [{
        Array whose dimensions are given by the `sizes` operands, and whose
        elements are all equal to `value`. If `value` is an array, its
        dimensions are appended to the given ones.
    }];

    let arguments = (ins Modelica_Value:$value, Variadic<Modelica_Integer>:$sizes);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $value `,` $sizes attr-dict `:` functional-type(operands, results)
    }];
}

def Modelica_LinspaceOp : Modelica_Op<"linspace", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Equally spaced elements (`linspace`).";

    let description = [{
        Real vector of `n` elements equally spaced between `start` and `stop`.
    }];

    let arguments = (ins
        Modelica_Value:$start,
        Modelica_Value:$stop,
        Modelica_Integer:$n
    );

    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $start `,` $stop `,` $n attr-dict `:` functional-type(operands, results)
    }];
}

//===---------------------------------------------------------------------===//
// Iterator-based expressions
//===---------------------------------------------------------------------===//

def Modelica_ArrayForOp : Modelica_Op<"array_for", [
        RecursiveMemoryEffects,
        SingleBlock
    ]>
{
    let summary = "Array constructor with iterators (`{e for i in r}`).";

    let description = [{
        Array obtained by evaluating the body for each combination of values of
        the iterators. Each iterator takes the values of the corresponding
        1-dimensional range operand, and it is available in the body as a block
        argument.

        With multiple iterators, the last iterator varies along the outermost
        dimension: `{e for i in u, j in v}` is equivalent to
        `{{e for i in u} for j in v}`, and thus has shape `size(v) x size(u)`.

        Example:

        ```mlir
        %r = modelica.array_for (%i in %u : tensor<3x!modelica.int>, %j in %v : tensor<2x!modelica.int>) -> tensor<2x3x!modelica.int> {
            %0 = modelica.sub %i, %j : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.yield %0 : !modelica.int
        }
        ```
    }];

    let arguments = (ins Variadic<RankedTensorOf<[Modelica_Scalar]>>:$ranges);
    let results = (outs Modelica_Value:$result);
    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Modelica_ReductionOp : Modelica_Op<"reduction", [
        RecursiveMemoryEffects,
        SingleBlock
    ]>
{
    let summary = "Reduction expression (`sum(e for i in r)`).";

    let description = [{
        Reduction of the values obtained by evaluating the body for each
        combination of values of the iterators. Each iterator takes the values
        of the corresponding 1-dimensional range operand, and it is available in
        the body as a block argument.

        Example:

        ```mlir
        %r = modelica.reduction sum (%i in %u : tensor<10x!modelica.int>) -> !modelica.int {
            %0 = modelica.mul %i, %i : (!modelica.int, !modelica.int) -> !modelica.int
            modelica.yield %0 : !modelica.int
        }
        ```
    }];

    let arguments = (ins
        Modelica_ReductionKind:$kind,
        Variadic<RankedTensorOf<[Modelica_Scalar]>>:$ranges
    );

    let results = (outs Modelica_Value:$result);
    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Array dimension and conversion functions
//===---------------------------------------------------------------------===//

def Modelica_NDimsOp : Modelica_UnaryOp<"ndims", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Number of dimensions (`ndims`).";

    let description = [{
        Number of dimensions of the operand.
    }];

    let hasFolder = 1;
}

def Modelica_SizeOp : Modelica_Op<"size", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Size of the dimensions (`size`).";

    let description = [{
        If the zero-based `dim` operand is specified, the size of that dimension
        of the array. Otherwise, the vector of the sizes of all the dimensions.
    }];

    let arguments = (ins Modelica_Value:$array, Optional<Modelica_Integer>:$dim);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $array (`,` $dim^)? attr-dict `:` functional-type(operands, results)
    }];

    let hasFolder = 1;
}

def Modelica_ScalarOp : Modelica_UnaryOp<"scalar", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Single element of an array (`scalar`).";

    let description = [{
        The only element of an array whose dimensions all have size 1.
    }];

    let hasFolder = 1;
}

def Modelica_VectorOp : Modelica_UnaryOp<"vector", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Vector of all the elements (`vector`).";

    let description = [{
        Vector containing all the elements of the operand, which must have at
        most one dimension with size greater than 1.
    }];
}

def Modelica_MatrixOp : Modelica_UnaryOp<"matrix", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Two-dimensional array (`matrix`).";

    let description = [{
        Matrix containing the elements of the first two dimensions of the
        operand, whose other dimensions must have size 1.
    }];
}

//===---------------------------------------------------------------------===//
// Array reduction functions
//===---------------------------------------------------------------------===//

def Modelica_ArrayReduceOp : Modelica_Op<"array_reduce", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Reduction of all the elements of an array.";

    let description = [{
        The `min(A)`, `max(A)`, `sum(A)` and `product(A)` functions applied to
        an array.
    }];

    let arguments = (ins Modelica_ReductionKind:$kind, Modelica_Value:$array);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $kind $array attr-dict `:` functional-type(operands, results)
    }];

    let hasFolder = 1;
    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Matrix and vector algebra functions
//===---------------------------------------------------------------------===//

def Modelica_TransposeOp : Modelica_UnaryOp<"transpose", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Matrix transpose (`transpose`).";

    let description = [{
        Permutation of the first two dimensions of the operand.
    }];
}

def Modelica_OuterProductOp
    : Modelica_BinaryOp<"outer_product", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Vector outer product (`outerProduct`).";

    let description = [{
        Outer product of two vectors.
    }];
}

def Modelica_SymmetricOp : Modelica_UnaryOp<"symmetric", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Symmetric matrix (`symmetric`).";

    let description = [{
        Symmetric matrix that is identical to the square matrix operand on and
        above the diagonal.
    }];
}

def Modelica_CrossOp : Modelica_BinaryOp<"cross", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Cross product (`cross`).";

    let description = [{
        Cross product of two 3-vectors.
    }];
}

def Modelica_SkewOp : Modelica_UnaryOp<"skew", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Skew symmetric matrix (`skew`).";

    let description = [{
        3x3 skew symmetric matrix associated with a 3-vector.
    }];
}

//===---------------------------------------------------------------------===//
// Numeric functions and conversion operators
//===---------------------------------------------------------------------===//

def Modelica_AbsOp : Modelica_UnaryOp<"abs", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Absolute value (`abs`).";
    let description = [{ Absolute value. }];
    let hasFolder = 1;
}

def Modelica_SignOp : Modelica_UnaryOp<"sign", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Sign (`sign`).";
    let description = [{ Sign of the operand, as an Integer. }];
    let hasFolder = 1;
}

def Modelica_SqrtOp : Modelica_UnaryOp<"sqrt", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Square root (`sqrt`).";
    let description = [{ Square root. }];
    let hasFolder = 1;
}

def Modelica_NthRootOp : Modelica_Op<"nth_root", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "N-th root (`nthRoot`).";

    let description = [{
        Real `n`-th root of the operand.
    }];

    let arguments = (ins Modelica_Value:$operand, Modelica_Integer:$n);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $operand `,` $n attr-dict `:` functional-type(operands, results)
    }];

    let hasFolder = 1;
}

def Modelica_EnumToIntegerOp
    : Modelica_UnaryOp<"enum_to_integer", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Enumeration to Integer conversion (`Integer(e)`).";

    let description = [{
        Ordinal value of an enumeration literal, starting from 1.
    }];

    let hasFolder = 1;
}

def Modelica_IntegerToEnumOp : Modelica_Op<"integer_to_enum", [NoMemoryEffect]>
{
    let summary = "Integer to enumeration conversion (`E(i)`).";

    let description = [{
        Enumeration literal having the given ordinal value, starting from 1. The
        enumeration type is given by the result type.
    }];

    let arguments = (ins Modelica_Value:$operand);
    let results = (outs Modelica_Value:$result);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

def Modelica_ToStringOp : Modelica_Op<"to_string", [
        NoMemoryEffect,
        AttrSizedOperandSegments
    ]>
{
    let summary = "Conversion to String (`String`).";

    let description = [{
        String representation of a Boolean, Integer, Real or enumeration scalar.
        The optional operands correspond to the homonymous named arguments of
        the operator.
    }];

    let arguments = (ins
        Modelica_Scalar:$value,
        Optional<Modelica_Integer>:$minimumLength,
        Optional<Modelica_Boolean>:$leftJustified,
        Optional<Modelica_Integer>:$significantDigits,
        Optional<Modelica_String>:$format
    );

    let results = (outs Modelica_String:$result);

    let assemblyFormat = [{
        $value
        (`minimumLength` $minimumLength^)?
        (`leftJustified` $leftJustified^)?
        (`significantDigits` $significantDigits^)?
        (`format` $format^)?
        attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Event-triggering mathematical functions
//===---------------------------------------------------------------------===//

def Modelica_DivTruncOp : Modelica_BinaryOp<"div_trunc", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Division with truncation toward zero (`div`).";

    let description = [{
        Algebraic quotient with any fractional part discarded. The result is
        Integer if both the operands are Integer, and Real otherwise.
    }];

    let hasFolder = 1;
}

def Modelica_ModOp : Modelica_BinaryOp<"mod", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Integer modulus (`mod`).";

    let description = [{
        `x - floor(x / y) * y`. The result is Integer if both the operands are
        Integer, and Real otherwise.
    }];

    let hasFolder = 1;
}

def Modelica_RemOp : Modelica_BinaryOp<"rem", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Integer remainder (`rem`).";

    let description = [{
        `x - div(x, y) * y`. The result is Integer if both the operands are
        Integer, and Real otherwise.
    }];

    let hasFolder = 1;
}

def Modelica_CeilOp : Modelica_UnaryOp<"ceil", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Ceiling (`ceil`).";
    let description = [{ Smallest integer Real not less than the operand. }];
    let hasFolder = 1;
}

def Modelica_FloorOp : Modelica_UnaryOp<"floor", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Floor (`floor`).";
    let description = [{ Largest integer Real not greater than the operand. }];
    let hasFolder = 1;
}

def Modelica_IntegerOp : Modelica_UnaryOp<"integer", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Conversion to Integer (`integer`).";
    let description = [{ Largest Integer not greater than the operand. }];
    let hasFolder = 1;
}

def Modelica_DelayOp : Modelica_Op<"delay", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Time delay (`delay`).";

    let description = [{
        Value of `expr` at time `time - delayTime`.
    }];

    let arguments = (ins
        Modelica_Value:$expr,
        Modelica_Value:$delayTime,
        Optional<Modelica_Value>:$delayMax
    );
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $expr `,` $delayTime (`,` $delayMax^)? attr-dict
        `:` functional-type(operands, results)
    }];
}

//===---------------------------------------------------------------------===//
// Elementary mathematical functions
//===---------------------------------------------------------------------===//

class Modelica_ElementaryOp<string mnemonic, string name>
    : Modelica_UnaryOp<mnemonic, [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Elementary mathematical function (`" # name # "`).";

    let description = [{
        Element-wise elementary mathematical function. The result is always
        Real.
    }];

    let hasFolder = 1;
}

def Modelica_SinOp : Modelica_ElementaryOp<"sin", "sin">;
def Modelica_CosOp : Modelica_ElementaryOp<"cos", "cos">;
def Modelica_TanOp : Modelica_ElementaryOp<"tan", "tan">;
def Modelica_AsinOp : Modelica_ElementaryOp<"asin", "asin">;
def Modelica_AcosOp : Modelica_ElementaryOp<"acos", "acos">;
def Modelica_AtanOp : Modelica_ElementaryOp<"atan", "atan">;
def Modelica_SinhOp : Modelica_ElementaryOp<"sinh", "sinh">;
def Modelica_CoshOp : Modelica_ElementaryOp<"cosh", "cosh">;
def Modelica_TanhOp : Modelica_ElementaryOp<"tanh", "tanh">;
def Modelica_ExpOp : Modelica_ElementaryOp<"exp", "exp">;
def Modelica_LogOp : Modelica_ElementaryOp<"log", "log">;
def Modelica_Log10Op : Modelica_ElementaryOp<"log10", "log10">;

def Modelica_Atan2Op : Modelica_BinaryOp<"atan2", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Four-quadrant arc tangent (`atan2`).";

    let description = [{
        Element-wise arc tangent of `lhs / rhs`, using the signs of the operands
        to determine the quadrant.
    }];

    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// Derivative and special purpose operators
//===---------------------------------------------------------------------===//

def Modelica_DerOp : Modelica_UnaryOp<"der", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Time derivative (`der`).";
    let description = [{ Time derivative of a Real scalar or array. }];
}

def Modelica_HomotopyOp : Modelica_Op<"homotopy", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Homotopy (`homotopy`).";

    let description = [{
        Homotopy between the `actual` and `simplified` expressions, used for
        solving initialization problems.
    }];

    let arguments = (ins Modelica_Value:$actual, Modelica_Value:$simplified);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $actual `,` $simplified attr-dict `:` functional-type(operands, results)
    }];
}

def Modelica_SemiLinearOp : Modelica_VariadicOp<"semi_linear", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Semi-linear function (`semiLinear`).";

    let description = [{
        `semiLinear(x, k_pos, k_neg)`, with operands `x`, `k_pos` and `k_neg` in
        this order.
    }];
}

def Modelica_TimeOp : Modelica_NullaryOp<"time", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Built-in variable `time`.";
    let description = [{ The independent variable. }];
}

//===---------------------------------------------------------------------===//
// Event-related operators
//===---------------------------------------------------------------------===//

def Modelica_InitialOp : Modelica_NullaryOp<"initial", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Initialization phase (`initial`).";
    let description = [{ True during the initialization phase. }];
}

def Modelica_TerminalOp : Modelica_NullaryOp<"terminal", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "End of the analysis (`terminal`).";
    let description = [{ True at the end of a successful analysis. }];
}

def Modelica_NoEventOp : Modelica_UnaryOp<"no_event", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Event inhibition (`noEvent`).";

    let description = [{
        The operand, with Real elementary relations taken literally, without
        triggering events.
    }];

    let hasFolder = 1;
}

def Modelica_SmoothOp : Modelica_Op<"smooth", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Smoothness annotation (`smooth`).";

    let description = [{
        The `expr` operand, declared to be `p` times continuously
        differentiable.
    }];

    let arguments = (ins Modelica_Integer:$p, Modelica_Value:$expr);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $p `,` $expr attr-dict `:` functional-type(operands, results)
    }];

    let hasFolder = 1;
}

def Modelica_SampleOp : Modelica_BinaryOp<"sample", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Periodic sampling (`sample`).";

    let description = [{
        True at the time instants `start + i * interval`. The clocked variants
        of the operator are not represented by this operation.
    }];
}

def Modelica_PreOp : Modelica_UnaryOp<"pre", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Left limit (`pre`).";
    let description = [{ Left limit of the operand at an event instant. }];
}

def Modelica_EdgeOp : Modelica_UnaryOp<"edge", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Rising edge (`edge`).";
    let description = [{ `b and not pre(b)`. }];
}

def Modelica_ChangeOp : Modelica_UnaryOp<"change", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Value change (`change`).";
    let description = [{ `v <> pre(v)`. }];
}

//===---------------------------------------------------------------------===//
// Binary min and max
//===---------------------------------------------------------------------===//

def Modelica_MinOp : Modelica_BinaryOp<"min", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Least of two scalars (`min(x, y)`).";

    let description = [{
        Least of two scalars, applied element-wise to arrays. For the reduction
        of an array, see `modelica.array_reduce`.
    }];

    let hasFolder = 1;
}

def Modelica_MaxOp : Modelica_BinaryOp<"max", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Greatest of two scalars (`max(x, y)`).";

    let description = [{
        Greatest of two scalars, applied element-wise to arrays. For the
        reduction of an array, see `modelica.array_reduce`.
    }];

    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// Component references
//
// A component reference is represented by a chain of operations: a
// 'modelica.ref', followed by any number of 'modelica.member' and
// 'modelica.subscript' operations. The chain evaluates to the value of the
// referenced component, but its structure is also used by the operations
// requiring a reference rather than a value, such as 'modelica.connect' and
// the destinations of assignments.
//===---------------------------------------------------------------------===//

def Modelica_RefOp : Modelica_Op<"ref", [Pure]>
{
    let summary = "Reference to a component.";

    let description = [{
        Reference to a component by means of its name. A simple name refers to
        the component or class constant with the given name, which is looked up
        starting from the scope enclosing the operation. A composite name starts
        with a class name, and refers to an element of that class, such as a
        constant of a package. Global names are rooted at `"."`.

        The component references used by the connect-equations and as
        destinations of the assignments always start with a simple name.

        Example:

        ```mlir
        %0 = modelica.ref @x : tensor<3x!modelica.real>
        %1 = modelica.ref @Modelica::@Constants::@pi : !modelica.real
        %2 = modelica.ref @"."::@Modelica::@Constants::@e : !modelica.real
        ```
    }];

    let arguments = (ins SymbolRefAttr:$name);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = "$name attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins
            "::mlir::Type":$resultType,
            "::llvm::StringRef":$name
        ), [{
            build($_builder, $_state, resultType,
                  ::mlir::FlatSymbolRefAttr::get($_builder.getContext(), name));
        }]>
    ];

    let extraClassDeclaration = [{
        /// Whether the name is made of a single identifier.
        bool hasSimpleName() {
            return getName().getNestedReferences().empty();
        }

        /// Get the identifier of a simple name.
        ::llvm::StringRef getSimpleName() {
            assert(hasSimpleName() && "The name is not simple");
            return getName().getRootReference().getValue();
        }
    }];
}

def Modelica_MemberOp : Modelica_Op<"member", [Pure]>
{
    let summary = "Member access (`a.b`).";

    let description = [{
        Access to a member of an object. If the operand is an array of objects,
        the operation is a slice operation and the dimensions of the operand are
        prepended to the ones of the member.

        Example:

        ```mlir
        %0 = modelica.member %r, @n : (tensor<10x!modelica.object<@Resistor>>) -> tensor<10x!modelica.object<@Pin>>
        ```
    }];

    let arguments = (ins Modelica_Value:$object, FlatSymbolRefAttr:$member);
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $object `,` $member attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

def Modelica_SubscriptOp : Modelica_Op<"subscript", [Pure]>
{
    let summary = "Array indexing (`a[i, :, v]`).";

    let description = [{
        Indexing of an array. Each subscript is either a colon, selecting a
        whole dimension, or an index. Scalar indices remove the corresponding
        dimension, while vector indices select the given elements. Indices can
        be Integer values, or Boolean and enumeration values for dimensions
        indexed by such types. Integer indices are zero-based, while Boolean and
        enumeration indices select the position of the value, with `false`
        before `true` and the literals in their order of declaration. If fewer
        subscripts than dimensions are given, the trailing dimensions are
        selected entirely.

        Example:

        ```mlir
        %0 = modelica.subscript %a[%i, :] : (tensor<3x4x!modelica.real>, !modelica.int) -> tensor<4x!modelica.real>
        ```
    }];

    let arguments = (ins
        RankedTensorOf<[Modelica_Scalar]>:$array,
        Variadic<Modelica_Value>:$indices,
        DenseBoolArrayAttr:$colons
    );

    let results = (outs Modelica_Value:$result);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins
            "::mlir::Type":$resultType,
            "::mlir::Value":$array,
            "::mlir::ValueRange":$indices
        )>
    ];
}

def Modelica_CallOp : Modelica_Op<"call">
{
    let summary = "Function call.";

    let description = [{
        Call to a function, including record constructors. The callee is the
        name of the function as written in the source code. Named arguments
        follow the positional ones.

        Example:

        ```mlir
        %0:2 = modelica.call @f(%a, y = %b) : (!modelica.real, !modelica.int) -> (!modelica.real, !modelica.real)
        ```
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<Modelica_Value>:$args,
        OptionalAttr<StrArrayAttr>:$argNames
    );

    let results = (outs Variadic<Modelica_Value>:$results);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        /// Get the number of positional arguments.
        size_t getNumPositionalArgs();
    }];
}

//===---------------------------------------------------------------------===//
// Operators on connectors
//===---------------------------------------------------------------------===//

def Modelica_InStreamOp : Modelica_UnaryOp<"in_stream", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Stream value close to the connection point (`inStream`).";

    let description = [{
        Value of a stream variable close to the connection point, assuming that
        the flow is from the connection point into the component. The operand
        must be a component reference.
    }];

    let hasVerifier = 1;
}

def Modelica_ActualStreamOp
    : Modelica_UnaryOp<"actual_stream", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Actual value of a stream variable (`actualStream`).";

    let description = [{
        Actual value of a stream variable for any flow direction. The operand
        must be a component reference.
    }];

    let hasVerifier = 1;
}

def Modelica_CardinalityOp
    : Modelica_UnaryOp<"cardinality", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Number of connections (`cardinality`).";

    let description = [{
        Number of times the connector is connected. The operator is deprecated,
        but still used in libraries. The operand must be a component reference.
    }];

    let hasVerifier = 1;
}

def Modelica_GetInstanceNameOp
    : Modelica_NullaryOp<"get_instance_name", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Name of the enclosing instance (`getInstanceName`).";

    let description = [{
        The name of the simulated model, followed by the name of the instance in
        which the operator is used.
    }];
}

def Modelica_SpatialDistributionOp : Modelica_Op<"spatial_distribution", [
        NoMemoryEffect,
        AttrSizedOperandSegments
    ]>
{
    let summary = "Variable-speed transport (`spatialDistribution`).";

    let description = [{
        Approximation of variable-speed transport of properties. The optional
        operands default to `{0.0, 1.0}` and `{0.0, 0.0}` respectively.
    }];

    let arguments = (ins
        Modelica_Value:$in0,
        Modelica_Value:$in1,
        Modelica_Value:$x,
        Modelica_Boolean:$positiveVelocity,
        Optional<Modelica_Value>:$initialPoints,
        Optional<Modelica_Value>:$initialValues
    );

    let results = (outs Modelica_Real:$out0, Modelica_Real:$out1);

    let assemblyFormat = [{
        $in0 `,` $in1 `,` $x `,` $positiveVelocity
        (`initialPoints` $initialPoints^)?
        (`initialValues` $initialValues^)?
        attr-dict `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

def Modelica_ConnectionsIsRootOp
    : Modelica_UnaryOp<"connections_is_root", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Whether a node is a root (`Connections.isRoot`).";

    let description = [{
        True if the overdetermined type or record is selected as root of the
        virtual connection graph.
    }];

    let hasVerifier = 1;
}

def Modelica_ConnectionsRootedOp
    : Modelica_UnaryOp<"connections_rooted", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Whether a node is closer to the root (`Connections.rooted`).";

    let description = [{
        True if the overdetermined type or record is closer to the root of the
        spanning tree than the other node of its branch. It also represents the
        deprecated `rooted` operator.
    }];

    let hasVerifier = 1;
}

def Modelica_ConnectionsUniqueRootIndicesOp
    : Modelica_Op<"connections_unique_root_indices", [
        NoMemoryEffect,
        InferTypeOpAdaptorWithIsCompatible
    ]>
{
    let summary = "Indices of unique roots (`Connections.uniqueRootIndices`).";

    let description = [{
        Permutation of the unique roots `roots` such that there is a path from
        each of the first `size(nodes, 1)` of them to the corresponding element
        of `nodes`.
    }];

    let arguments = (ins
        Modelica_Value:$roots,
        Modelica_Value:$nodes,
        Modelica_String:$message
    );
    let results = (outs Modelica_Value:$result);

    let assemblyFormat = [{
        $roots `,` $nodes `,` $message attr-dict
        `:` functional-type(operands, results)
    }];

    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Elements of classes
//
// The operations in this section represent the class tree, that is the content
// of the classes as written in the source code. Names of classes and
// components are kept as written: they are resolved according to the lookup
// rules of the language only during instantiation.
//
// Symbol names are the Modelica identifiers. Quoted identifiers keep their
// quotes, as they are part of the identifier. Global
// names are represented by symbol references whose root
// is the reserved name ".", which is not a valid Modelica identifier.
//
// Classes and enumerations are symbol tables, so that the uniqueness of the
// names of their elements is verified and absolute paths
// can be resolved with the MLIR symbol utilities. The elements themselves
// have a 'sym_name' attribute but don't implement the Symbol interface, as
// they can also be declared within modifications, whose operations are not
// symbol tables.
//===---------------------------------------------------------------------===//

// Prefixes that can be applied to the declaration of an element.
// 'each' is allowed only within modifications.
defvar Modelica_ElementPrefixes = (ins
    UnitAttr:$isProtected,
    UnitAttr:$isRedeclare,
    UnitAttr:$isEach,
    UnitAttr:$isFinal,
    UnitAttr:$isInner,
    UnitAttr:$isOuter,
    UnitAttr:$isReplaceable
);

// Documentation of an element.
defvar Modelica_ElementDocumentation = (ins
    OptionalAttr<StrAttr>:$description,
    OptionalAttr<DictionaryAttr>:$annotation
);

// Operations that can contain elements declared as part of a modification.
defvar Modelica_ElementParents = [
    "::mlir::ModuleOp", "ClassOp", "ComponentOp", "ExtendsOp", "ModifyOp",
    "ConstrainedByOp"];

//===---------------------------------------------------------------------===//
// Classes
//===---------------------------------------------------------------------===//

def Modelica_ClassOp : Modelica_Op<"class", [
        SymbolTable,
        IsolatedFromAbove,
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_ElementParents>
    ]>
{
    let summary = "Class definition.";

    let description = [{
        The operation represents the definition of a class. The body contains
        the elements of the class: nested classes, components, extends-clauses,
        import-clauses, equation and algorithm sections, and the external
        function clause.

        The forms of class definitions are represented as follows:
        - long class definitions contain their elements;
        - short class definitions, marked with `short`, contain a single
          `modelica.extends`, which may carry array dimensions and an input /
          output prefix;
        - class-extends redeclarations, marked with `extends`, contain a
          `modelica.extends` without base class, whose modification applies to
          the inherited class with the same name;
        - derivative definitions (`function df = der(f, x)`), marked with `der`,
          have an empty body.

        The constraining clause of a replaceable class is represented by a
        `modelica.constrainedby` operation in the body.

        Example:

        ```mlir
        modelica.class partial model @TwoPin {
            modelica.component @p : @Pin -> !modelica.object<@Pin>
            modelica.component @n : @Pin -> !modelica.object<@Pin>
        }

        modelica.class short type @Voltage {
            modelica.extends @Real {
                modelica.modify @unit {
                    %0 = modelica.constant #modelica<string "V"> : !modelica.string
                    modelica.bind %0 : !modelica.string
                }
            }
        }
        ```
    }];

    let arguments = !con(
        (ins SymbolNameAttr:$sym_name, Modelica_ClassKind:$kind),
        Modelica_ElementPrefixes,
        (ins
            UnitAttr:$isEncapsulated,
            UnitAttr:$isPartial,
            UnitAttr:$isImpure,
            UnitAttr:$isShort,
            UnitAttr:$isClassExtends,
            OptionalAttr<SymbolRefAttr>:$derivativeOf,
            OptionalAttr<StrArrayAttr>:$derivativeArgs
        ),
        Modelica_ElementDocumentation
    );

    let regions = (region SizedRegion<1>:$bodyRegion);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name, "ClassKind":$kind)>
    ];

    let extraClassDeclaration = [{
        ::mlir::Block *getBody() { return &getBodyRegion().front(); }
    }];
}

def Modelica_EnumerationOp : Modelica_Op<"enumeration", [
        SymbolTable,
        IsolatedFromAbove,
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_ElementParents>
    ]>
{
    let summary = "Enumeration type definition.";

    let description = [{
        The operation represents the definition of an enumeration type, whose
        literals are represented by the `modelica.enum_literal` operations
        contained in the body. The `unspecified` keyword represents
        `enumeration(:)`.
    }];

    let arguments = !con(
        (ins SymbolNameAttr:$sym_name),
        Modelica_ElementPrefixes,
        (ins UnitAttr:$isUnspecified),
        Modelica_ElementDocumentation
    );

    let regions = (region SizedRegion<1>:$bodyRegion);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::Block *getBody() { return &getBodyRegion().front(); }
    }];
}

def Modelica_EnumLiteralOp : Modelica_Op<"enum_literal", [
        HasParent<"EnumerationOp">
    ]>
{
    let summary = "Enumeration literal.";

    let description = [{
        A literal of an enumeration type. Its ordinal value is given by its
        position within the enumeration, starting from 1.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$description
    );

    let assemblyFormat = "$sym_name ($description^)? attr-dict";
}

def Modelica_ConstrainedByOp : Modelica_Op<"constrainedby", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<["ClassOp", "EnumerationOp"]>
    ]>
{
    let summary = "Constraining clause of a replaceable class.";

    let description = [{
        The constraining type of the enclosing replaceable class, with its
        optional modification.
    }];

    let arguments = (ins SymbolRefAttr:$constrainingClass);
    let regions = (region MaxSizedRegion<1>:$modificationRegion);

    let assemblyFormat = [{
        $constrainingClass ($modificationRegion^)? attr-dict
    }];

    let hasRegionVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Components
//===---------------------------------------------------------------------===//

def Modelica_ComponentOp : Modelica_Op<"component", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_ElementParents>
    ]>
{
    let summary = "Component declaration.";

    let description = [{
        The operation represents the declaration of a component. The declaration
        refers to the class of the component by its name, as written in the
        source code, and also specifies the type of the values obtained when
        referring to the component. Such type takes into account the array
        dimensions, and is determined from the constraining type in case of
        replaceable classes.

        The operation has four optional regions, printed in this order:
        - the modification, containing `modelica.modify` operations,
          redeclarations of elements, and an optional `modelica.bind` for the
          declaration equation.
        - `dims`: the array dimensions, in the order of the type of the
          component, that is the dimensions of the declaration followed by the
          ones of the type specifier. The region yields one Integer value for
          each dimension of kind `?`.
        - `if`: the condition of a conditional component, yielding a Boolean
          value.
        - `constrainedby`: the modification of the constraining type of a
          replaceable component.

        The expressions within the regions are evaluated in the scope enclosing
        the declaration.

        Example:

        ```mlir
        modelica.component parameter @R : @Real -> !modelica.real {
            %0 = modelica.constant #modelica<real 1.0> : !modelica.real
            modelica.bind %0 : !modelica.real
        }

        modelica.component @r : @Resistor -> tensor<?x!modelica.object<@Resistor>> {
            modelica.modify each @R {
                %0 = modelica.constant #modelica<real 2.0> : !modelica.real
                modelica.bind %0 : !modelica.real
            }
        } dims [?] {
            %n = modelica.ref @N : !modelica.int
            modelica.yield %n : !modelica.int
        }
        ```
    }];

    let arguments = !con(
        (ins
            SymbolNameAttr:$sym_name,
            SymbolRefAttr:$className,
            TypeAttr:$type
        ),
        Modelica_ElementPrefixes,
        (ins
            OptionalAttr<Modelica_ConnectorKind>:$connectorKind,
            OptionalAttr<Modelica_Variability>:$variability,
            OptionalAttr<Modelica_Causality>:$causality,
            DefaultValuedAttr<TypedArrayAttrBase<Modelica_DimensionAttr, "dimensions">, "{}">:$dimensions,
            OptionalAttr<SymbolRefAttr>:$constrainingClass
        ),
        Modelica_ElementDocumentation
    );

    let regions = (region
        MaxSizedRegion<1>:$dimensionsRegion,
        MaxSizedRegion<1>:$modificationRegion,
        MaxSizedRegion<1>:$conditionRegion,
        MaxSizedRegion<1>:$constraintRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::StringRef":$name,
            "::mlir::SymbolRefAttr":$className,
            "::mlir::Type":$type
        )>
    ];
}

//===---------------------------------------------------------------------===//
// Extends and import clauses
//===---------------------------------------------------------------------===//

def Modelica_ExtendsOp : Modelica_Op<"extends", [
        NoTerminator,
        SingleBlock,
        HasParent<"ClassOp">
    ]>
{
    let summary = "Extends-clause.";

    let description = [{
        The operation represents an extends-clause. The modification region
        contains the modifications of the base class, including the
        redeclarations and the deselections of selective model extension.

        Within the body of a class-extends redeclaration, the base class is
        omitted and the operation refers to the inherited class with the same
        name of the enclosing one.

        Within the body of a short class definition, the operation may also
        specify an input / output prefix and array dimensions, with the same
        representation used by components.
    }];

    let arguments = (ins
        OptionalAttr<SymbolRefAttr>:$baseClass,
        UnitAttr:$isProtected,
        OptionalAttr<Modelica_Causality>:$causality,
        DefaultValuedAttr<TypedArrayAttrBase<Modelica_DimensionAttr, "dimensions">, "{}">:$dimensions,
        OptionalAttr<DictionaryAttr>:$annotation
    );

    let regions = (region
        MaxSizedRegion<1>:$dimensionsRegion,
        MaxSizedRegion<1>:$modificationRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

def Modelica_ImportOp : Modelica_Op<"import", [HasParent<"ClassOp">]>
{
    let summary = "Import-clause.";

    let description = [{
        The operation represents an import-clause, in one of its forms:

        ```mlir
        // import A.B;
        modelica.import @A::@B
        // import X = A.B;
        modelica.import @X = @A::@B
        // import A.B.*;
        modelica.import @A::@B unqualified
        // import A.B.{C, D};
        modelica.import @A::@B {@C, @D}
        ```
    }];

    let arguments = !con(
        (ins
            SymbolRefAttr:$path,
            OptionalAttr<FlatSymbolRefAttr>:$alias,
            UnitAttr:$isUnqualified,
            OptionalAttr<FlatSymbolRefArrayAttr>:$names,
            UnitAttr:$isProtected
        ),
        Modelica_ElementDocumentation
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Modifications
//===---------------------------------------------------------------------===//

def Modelica_ModifyOp : Modelica_Op<"modify", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<["ComponentOp", "ExtendsOp", "ModifyOp", "ConstrainedByOp",
                     "VariableOp"]>
    ]>
{
    let summary = "Element modification.";

    let description = [{
        Modification of an element of the class being modified. The name may be
        composite, as in `a.b.c = 1`. The region contains the nested
        modifications, redeclarations, and the optional `modelica.bind` for the
        value of the element. The expressions within the region are evaluated in
        the scope where the modification is written.

        Example:

        ```mlir
        // heatPort(T(start = 300))
        modelica.modify @heatPort {
            modelica.modify @T {
                modelica.modify @start {
                    %0 = modelica.constant #modelica<real 300.0> : !modelica.real
                    modelica.bind %0 : !modelica.real
                }
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolRefAttr:$name,
        UnitAttr:$isEach,
        UnitAttr:$isFinal,
        OptionalAttr<StrAttr>:$description
    );

    let regions = (region MaxSizedRegion<1>:$modificationRegion);

    let assemblyFormat = [{
        (`each` $isEach^)? (`final` $isFinal^)? $name
        ($modificationRegion^)? attr-dict
    }];

    let hasRegionVerifier = 1;
}

def Modelica_BindOp : Modelica_Op<"bind", [
        ParentOneOf<["ComponentOp", "ModifyOp", "VariableOp"]>
    ]>
{
    let summary = "Value of a modification.";

    let description = [{
        The value given to the modified element (`= expression`), or the removal
        of the value of the modified element (`= break`).
    }];

    let arguments = (ins Optional<Modelica_Value>:$value, UnitAttr:$isBreak);

    let assemblyFormat = [{
        (`break` $isBreak^)? ($value^ `:` type($value))? attr-dict
    }];

    let hasVerifier = 1;
}

def Modelica_DeselectOp : Modelica_Op<"deselect", [HasParent<"ExtendsOp">]>
{
    let summary = "Component deselection (`break name`).";

    let description = [{
        Deselection of a component of the base class, together with all the
        connections involving it.
    }];

    let arguments = (ins FlatSymbolRefAttr:$component);
    let assemblyFormat = "$component attr-dict";
}

def Modelica_DeselectConnectOp : Modelica_Op<"deselect_connect", [
        SingleBlock,
        HasParent<"ExtendsOp">
    ]>
{
    let summary = "Connection deselection (`break connect(a, b)`).";

    let description = [{
        Deselection of the connections of the base class that are syntactically
        equivalent to the given one. The region yields the two component
        references.
    }];

    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
    let hasRegionVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Sections
//===---------------------------------------------------------------------===//

def Modelica_EquationSectionOp : Modelica_Op<"equations", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<["ClassOp", "ModelOp"]>
    ]>
{
    let summary = "Equation section.";

    let description = [{
        An equation section, optionally of initial equations.
    }];

    let arguments = (ins UnitAttr:$isInitial);
    let regions = (region SizedRegion<1>:$bodyRegion);

    let assemblyFormat = "(`initial` $isInitial^)? $bodyRegion attr-dict";
}

def Modelica_AlgorithmSectionOp : Modelica_Op<"algorithm", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<["ClassOp", "ModelOp", "FunctionOp"]>
    ]>
{
    let summary = "Algorithm section.";

    let description = [{
        An algorithm section, optionally of initial statements.
    }];

    let arguments = (ins UnitAttr:$isInitial);
    let regions = (region SizedRegion<1>:$bodyRegion);

    let assemblyFormat = "(`initial` $isInitial^)? $bodyRegion attr-dict";
}

def Modelica_ExternalOp : Modelica_Op<"external", [
        SingleBlock,
        ParentOneOf<["ClassOp", "FunctionOp"]>
    ]>
{
    let summary = "External function clause.";

    let description = [{
        The external clause of a function. If the name of the external function
        is specified, the region yields the arguments of the external call,
        preceded by the component reference receiving the result if `output` is
        specified. Without a function name, the default mapping of the arguments
        applies and the region is empty.

        Example:

        ```mlir
        // external "C" y = f(x, size(x, 1)) annotation(Library = "f");
        modelica.external "C" "f" output {
            %y = modelica.ref @y : !modelica.real
            %x = modelica.ref @x : tensor<?x!modelica.real>
            %one = modelica.constant #modelica<int 1> : !modelica.int
            %n = modelica.size %x, %one : (tensor<?x!modelica.real>, !modelica.int) -> !modelica.int
            modelica.yield %y, %x, %n : !modelica.real, tensor<?x!modelica.real>, !modelica.int
        } {annotation = {Library = "f"}}
        ```
    }];

    let arguments = (ins
        StrAttr:$language,
        OptionalAttr<StrAttr>:$function,
        UnitAttr:$hasOutput,
        OptionalAttr<DictionaryAttr>:$annotation
    );

    let regions = (region MaxSizedRegion<1>:$bodyRegion);

    let assemblyFormat = [{
        $language ($function^)? (`output` $hasOutput^)? ($bodyRegion^)?
        attr-dict
    }];

    let hasVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Equations and statements
//
// Each equation and statement owns the expressions it uses: they are computed
// within its regions, and passed to it by means of terminators. This way,
// the component references within an algorithm are evaluated when the
// statement using them is executed.
//===---------------------------------------------------------------------===//

defvar Modelica_EquationParents = [
    "EquationSectionOp", "ForEquationOp", "IfEquationOp", "WhenEquationOp"];

defvar Modelica_StatementParents = [
    "AlgorithmSectionOp", "ForOp", "IfOp", "WhileOp", "WhenOp"];

// Operation whose only region computes the operands of the operation, and is
// terminated by a 'modelica.yield'.
class Modelica_YieldingOp<string mnemonic, list<string> parents,
                          list<Trait> traits = []>
    : Modelica_Op<mnemonic, !listconcat(traits, [
        SingleBlock,
        ParentOneOf<parents>
    ])>
{
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
    let hasRegionVerifier = 1;
}

//===---------------------------------------------------------------------===//
// Equations
//===---------------------------------------------------------------------===//

def Modelica_EquationOp : Modelica_Op<"equation", [
        SingleBlock,
        ParentOneOf<Modelica_EquationParents>
    ]>
{
    let summary = "Simple equality equation.";

    let description = [{
        A simple equality equation. The body computes the two sides, which are
        equated by the `modelica.equation_sides` terminator. Equations involving
        multiple outputs of a function call have multiple values on both sides;
        outputs omitted in the source code (e.g., `(a, , b) = f(x)`) are omitted
        also from the right-hand side.
    }];

    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
    let hasRegionVerifier = 1;
}

def Modelica_EquationSidesOp : Modelica_Op<"equation_sides", [
        Terminator,
        AttrSizedOperandSegments,
        HasParent<"EquationOp">
    ]>
{
    let summary = "Sides of an equation.";

    let description = [{
        Terminator of `modelica.equation`, equating each value of the left-hand
        side with the corresponding value of the right-hand side. Integer and
        Real values can be equated.
    }];

    let arguments = (ins
        Variadic<Modelica_Value>:$lhs,
        Variadic<Modelica_Value>:$rhs
    );

    let assemblyFormat = [{
        $lhs `=` $rhs attr-dict `:` type($lhs) `=` type($rhs)
    }];

    let hasVerifier = 1;
}

def Modelica_CallEquationOp : Modelica_YieldingOp<
    "call_equation", Modelica_EquationParents>
{
    let summary = "Function call equation.";

    let description = [{
        A call to a function used as an equation, without using its outputs. The
        region yields the arguments of the call. Named arguments follow the
        positional ones, and their names are specified by `argNames`.
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        OptionalAttr<StrArrayAttr>:$argNames
    );

    let assemblyFormat = "$callee $bodyRegion attr-dict";
}

def Modelica_ForEquationOp : Modelica_Op<"for_equation", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_EquationParents>
    ]>
{
    let summary = "For-equation.";

    let description = [{
        A for-equation. The range region yields the vector of values taken by
        the loop variable, which is the argument of the body region. If the
        range region is empty, the iteration range is implicit and is to be
        deduced from the usage of the loop variable as array index.
        For-equations with multiple indices are represented as nested
        for-equations.
    }];

    let regions = (region
        MaxSizedRegion<1>:$rangeRegion,
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasRegionVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::BlockArgument getInductionVariable() {
            return getBodyRegion().front().getArgument(0);
        }

        bool hasImplicitRange() { return getRangeRegion().empty(); }
    }];
}

def Modelica_IfEquationOp : Modelica_Op<"if_equation", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_EquationParents>
    ]>
{
    let summary = "If-equation.";

    let description = [{
        An if-equation. The condition region yields a Boolean value. The
        `elseif` branches are represented by nesting another if-equation into
        the else region.
    }];

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$thenRegion,
        MaxSizedRegion<1>:$elseRegion
    );

    let assemblyFormat = [{
        $conditionRegion `then` $thenRegion (`else` $elseRegion^)? attr-dict
    }];

    let hasRegionVerifier = 1;
}

def Modelica_WhenEquationOp : Modelica_Op<"when_equation", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_EquationParents>
    ]>
{
    let summary = "When-equation.";

    let description = [{
        A when-equation. The condition region yields a Boolean scalar or vector.
        The `elsewhen` branches are represented by a single when-equation nested
        into the elsewhen region.
    }];

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$thenRegion,
        MaxSizedRegion<1>:$elseWhenRegion
    );

    let assemblyFormat = [{
        $conditionRegion `then` $thenRegion (`elsewhen` $elseWhenRegion^)?
        attr-dict
    }];

    let hasRegionVerifier = 1;
}

def Modelica_ConnectOp : Modelica_YieldingOp<
    "connect", Modelica_EquationParents>
{
    let summary = "Connect-equation.";

    let description = [{
        A connect-equation. The region yields the two connected component
        references, each of which is a chain of `modelica.ref`,
        `modelica.member` and `modelica.subscript` operations. Whether a
        connector is inside or outside can be determined from the structure of
        the chain.

        Example:

        ```mlir
        // connect(r.n, g.p)
        modelica.connect {
            %r = modelica.ref @r : !modelica.object<@Resistor>
            %rn = modelica.member %r, @n : (!modelica.object<@Resistor>) -> !modelica.object<@Pin>
            %g = modelica.ref @g : !modelica.object<@Ground>
            %gp = modelica.member %g, @p : (!modelica.object<@Ground>) -> !modelica.object<@Pin>
            modelica.yield %rn, %gp : !modelica.object<@Pin>, !modelica.object<@Pin>
        }
        ```

        Within instantiated models, the component references are absolute, and
        the structure of the chain doesn't tell anymore whether a connector is
        inside or outside. Such information is thus stored by the `lhsOutside`
        and `rhsOutside` flags, which are set for the connectors that are
        outside with respect to the instance whose class contains the
        connect-equation.
    }];

    let arguments = (ins UnitAttr:$lhsOutside, UnitAttr:$rhsOutside);
}

//===---------------------------------------------------------------------===//
// Operators usable both as equations and as statements
//===---------------------------------------------------------------------===//

def Modelica_ReinitOp : Modelica_YieldingOp<"reinit",
    !listconcat(Modelica_EquationParents, Modelica_StatementParents)>
{
    let summary = "Reinitialization (`reinit`).";

    let description = [{
        Reinitialization of a state variable at an event instant. The region
        yields the component reference to the variable and its new value.
    }];
}

def Modelica_AssertOp : Modelica_YieldingOp<"assert",
    !listconcat(Modelica_EquationParents, Modelica_StatementParents)>
{
    let summary = "Assertion (`assert`).";

    let description = [{
        Assertion. The region yields the condition, the message, and optionally
        the assertion level.
    }];
}

def Modelica_TerminateOp : Modelica_YieldingOp<"terminate",
    !listconcat(Modelica_EquationParents, Modelica_StatementParents)>
{
    let summary = "Successful termination (`terminate`).";

    let description = [{
        Successful termination of the analysis. The region yields the message.
    }];
}

//===---------------------------------------------------------------------===//
// Equation operators for overconstrained connections
//===---------------------------------------------------------------------===//

def Modelica_ConnectionsBranchOp : Modelica_YieldingOp<
    "connections_branch", Modelica_EquationParents>
{
    let summary = "Required spanning-tree edge (`Connections.branch`).";

    let description = [{
        The region yields the two component references of the overdetermined
        types or records.
    }];
}

def Modelica_ConnectionsRootOp : Modelica_YieldingOp<
    "connections_root", Modelica_EquationParents>
{
    let summary = "Definite root node (`Connections.root`).";

    let description = [{
        The region yields the component reference of the overdetermined type or
        record.
    }];
}

def Modelica_ConnectionsPotentialRootOp : Modelica_YieldingOp<
    "connections_potential_root", Modelica_EquationParents>
{
    let summary = "Potential root node (`Connections.potentialRoot`).";

    let description = [{
        The region yields the component reference of the overdetermined type or
        record, and optionally its Integer priority.
    }];
}

def Modelica_ConnectionsUniqueRootOp : Modelica_YieldingOp<
    "connections_unique_root", Modelica_EquationParents>
{
    let summary = "Unique root node (`Connections.uniqueRoot`).";

    let description = [{
        The region yields the component reference of the overdetermined type or
        record, and the message.
    }];
}

//===---------------------------------------------------------------------===//
// Statements
//===---------------------------------------------------------------------===//

def Modelica_AssignmentOp : Modelica_Op<"assignment", [
        SingleBlock,
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "Assignment statement.";

    let description = [{
        An assignment statement, including the assignment of multiple outputs of
        a function call. The body computes the destination component references
        and the values, which are assigned by the `modelica.assignment_sides`
        terminator. A destination may also be the derivative of a component
        reference (`der(x) := ...`).
    }];

    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
    let hasRegionVerifier = 1;
}

def Modelica_AssignmentSidesOp : Modelica_Op<"assignment_sides", [
        Terminator,
        AttrSizedOperandSegments,
        HasParent<"AssignmentOp">
    ]>
{
    let summary = "Destinations and values of an assignment.";

    let description = [{
        Terminator of `modelica.assignment`, assigning each value to the
        corresponding destination. Integer values can be assigned to Real
        destinations, except in case of multiple destinations.
    }];

    let arguments = (ins
        Variadic<Modelica_Value>:$destinations,
        Variadic<Modelica_Value>:$values
    );

    let assemblyFormat = [{
        $destinations `=` $values attr-dict
        `:` type($destinations) `=` type($values)
    }];

    let hasVerifier = 1;
}

def Modelica_CallStatementOp : Modelica_YieldingOp<
    "call_statement", Modelica_StatementParents>
{
    let summary = "Function call statement.";

    let description = [{
        A call to a function used as a statement, without using its outputs. The
        region yields the arguments of the call. Named arguments follow the
        positional ones, and their names are specified by `argNames`.
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        OptionalAttr<StrArrayAttr>:$argNames
    );

    let assemblyFormat = "$callee $bodyRegion attr-dict";
}

def Modelica_IfOp : Modelica_Op<"if", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "If-statement.";

    let description = [{
        An if-statement. The `elseif` branches are represented by nesting
        another if-statement into the else region.
    }];

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$thenRegion,
        MaxSizedRegion<1>:$elseRegion
    );

    let assemblyFormat = [{
        $conditionRegion `then` $thenRegion (`else` $elseRegion^)? attr-dict
    }];

    let hasRegionVerifier = 1;
}

def Modelica_ForOp : Modelica_Op<"for", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "For-statement.";

    let description = [{
        A for-statement, with the same structure of `modelica.for_equation`.
    }];

    let regions = (region
        MaxSizedRegion<1>:$rangeRegion,
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasRegionVerifier = 1;

    let extraClassDeclaration = [{
        ::mlir::BlockArgument getInductionVariable() {
            return getBodyRegion().front().getArgument(0);
        }

        bool hasImplicitRange() { return getRangeRegion().empty(); }
    }];
}

def Modelica_WhileOp : Modelica_Op<"while", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "While-statement.";

    let description = [{
        A while-statement. The condition region yields a Boolean value, and is
        evaluated before each iteration.
    }];

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$conditionRegion `do` $bodyRegion attr-dict";
    let hasRegionVerifier = 1;
}

def Modelica_WhenOp : Modelica_Op<"when", [
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "When-statement.";

    let description = [{
        A when-statement, with the same structure of `modelica.when_equation`.
    }];

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$thenRegion,
        MaxSizedRegion<1>:$elseWhenRegion
    );

    let assemblyFormat = [{
        $conditionRegion `then` $thenRegion (`elsewhen` $elseWhenRegion^)?
        attr-dict
    }];

    let hasRegionVerifier = 1;
}

def Modelica_BreakOp : Modelica_Op<"break", [
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "Break-statement.";

    let description = [{
        Termination of the innermost enclosing for or while loop.
    }];

    let assemblyFormat = "attr-dict";
    let hasVerifier = 1;
}

def Modelica_ReturnOp : Modelica_Op<"return", [
        ParentOneOf<Modelica_StatementParents>
    ]>
{
    let summary = "Return-statement.";

    let description = [{
        Termination of the current function call.
    }];

    let assemblyFormat = "attr-dict";
}

//===---------------------------------------------------------------------===//
// Instances
//
// The operations in this section represent the result of the instantiation of a
// class: the instance tree of a model, the flattened model, and the
// instantiated functions.
//
// The instance tree of a model is represented by a 'modelica.model' whose
// body contains nested 'modelica.instance' and 'modelica.variable'
// operations, mirroring the components of the instantiated class. Arrays of
// structured components are kept as single instances having an array type,
// so that the size of the IR doesn't depend on the number of elements.
// Inheritance, modifications, redeclarations, conditional components and
// inner / outer elements are resolved during instantiation.
//
// The equations and algorithms of all the instances are placed in the body
// of the model. The component references are absolute, that is they start
// from the elements of the model, and the equations of arrays of instances
// are enclosed in for-equations iterating over the indices of the elements.
//
// Flattening replaces the instances with their primitive variables, whose
// names are the dotted paths of the instance tree, and whose dimensions are
// the ones of the enclosing instances followed by their own ones.
//===---------------------------------------------------------------------===//

// Operations that can contain instances and variables.
defvar Modelica_InstanceParents = ["ModelOp", "InstanceOp", "FunctionOp"];

def Modelica_ModelOp : Modelica_Op<"model", [
        Symbol,
        SymbolTable,
        IsolatedFromAbove,
        NoTerminator,
        SingleBlock,
        HasParent<"::mlir::ModuleOp">
    ]>
{
    let summary = "Instantiated model.";

    let description = [{
        The instance of the class being simulated. The body contains the
        instance tree and the equations and algorithms of all the instances.

        Example:

        ```mlir
        modelica.model @Circuit {
            modelica.variable parameter @N : !modelica.int {
                %0 = modelica.constant #modelica<int 3> : !modelica.int
                modelica.bind %0 : !modelica.int
            }
            modelica.instance model @r : tensor<3x!modelica.object<@Resistor>> {
                modelica.variable parameter @R : !modelica.real {
                ^bb0(%i: !modelica.int):
                    ...
                }
                ...
            }
            modelica.equations {
                modelica.for_equation %i : !modelica.int in {
                    ...
                } do {
                    ...
                }
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$description,
        OptionalAttr<DictionaryAttr>:$annotation
    );

    let regions = (region SizedRegion<1>:$bodyRegion);

    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name), [{
            $_state.addAttribute(getSymNameAttrName($_state.name),
                                 $_builder.getStringAttr(name));
            $_state.addRegion()->emplaceBlock();
        }]>
    ];

    let extraClassDeclaration = [{
        ::mlir::Block *getBody() { return &getBodyRegion().front(); }
    }];
}

def Modelica_FunctionOp : Modelica_Op<"function", [
        Symbol,
        SymbolTable,
        IsolatedFromAbove,
        NoTerminator,
        SingleBlock,
        HasParent<"::mlir::ModuleOp">
    ]>
{
    let summary = "Instantiated function.";

    let description = [{
        The instance of a function class. The name is the dotted path of the
        function. The body contains the variables of the function, the algorithm
        sections, and the optional external clause. The dimensions of the
        variables may depend on the inputs, in which case they are computed by
        the dimensions region of the variables.

        Example:

        ```mlir
        modelica.function @"Modelica.Math.Vectors.norm" {
            modelica.variable input @v : tensor<?x!modelica.real>
            modelica.variable output @result : !modelica.real
            modelica.algorithm {
                ...
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        UnitAttr:$isImpure,
        OptionalAttr<StrAttr>:$description,
        OptionalAttr<DictionaryAttr>:$annotation
    );

    let regions = (region SizedRegion<1>:$bodyRegion);

    let assemblyFormat = [{
        (`impure` $isImpure^)? $sym_name attr-dict-with-keyword $bodyRegion
    }];

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name), [{
            $_state.addAttribute(getSymNameAttrName($_state.name),
                                 $_builder.getStringAttr(name));
            $_state.addRegion()->emplaceBlock();
        }]>
    ];

    let extraClassDeclaration = [{
        ::mlir::Block *getBody() { return &getBodyRegion().front(); }
    }];
}

def Modelica_InstanceOp : Modelica_Op<"instance", [
        Symbol,
        SymbolTable,
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_InstanceParents>
    ]>
{
    let summary = "Instance of a structured component.";

    let description = [{
        The instance of a component whose class is not a simple type, such as a
        model, a block, a connector or a record. The body contains the instances
        and variables of the elements of the class, after the application of
        inheritance, modifications and redeclarations.

        The type is the object type of the class of the instance, or a
        statically shaped array of such objects. In the latter case, the
        elements of the array share the same structure, and the elements of the
        instance are instantiated once for all of them.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        Modelica_ClassKind:$kind,
        UnitAttr:$isProtected,
        OptionalAttr<StrAttr>:$description
    );

    let regions = (region SizedRegion<1>:$bodyRegion);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::StringRef":$name,
            "::mlir::Type":$type,
            "ClassKind":$kind
        )>
    ];

    let extraClassDeclaration = [{
        ::mlir::Block *getBody() { return &getBodyRegion().front(); }

        /// Get the dimensions of the array of instances.
        ::llvm::ArrayRef<int64_t> getShape();
    }];
}

def Modelica_VariableOp : Modelica_Op<"variable", [
        Symbol,
        NoTerminator,
        SingleBlock,
        ParentOneOf<Modelica_InstanceParents>
    ]>
{
    let summary = "Primitive variable.";

    let description = [{
        An instance of a component of simple type, or an array of such
        components.

        The modification region contains the merged modification of the
        variable: the optional `modelica.bind` for the declaration equation, and
        the `modelica.modify` operations for the attributes of the type, such as
        `start` and `unit`. The block of the region has an Integer argument for
        each dimension of the enclosing instances, outermost first, which is the
        index of the element of the enclosing arrays of instances the
        modification applies to. Within models, the dimensions of variables are
        static. Within functions, dimensions may be dynamic, in which case the
        dimensions region yields their sizes.

        Example:

        ```mlir
        modelica.variable parameter @R : !modelica.real {
        ^bb0(%i: !modelica.int):
            modelica.modify @unit {
                %0 = modelica.constant #modelica<string "Ohm"> : !modelica.string
                modelica.bind %0 : !modelica.string
            }
            %0 = modelica.ref @Rs : tensor<3x!modelica.real>
            %1 = modelica.subscript %0[%i] : (tensor<3x!modelica.real>, !modelica.int) -> !modelica.real
            modelica.bind %1 : !modelica.real
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        OptionalAttr<Modelica_ConnectorKind>:$connectorKind,
        OptionalAttr<Modelica_Variability>:$variability,
        OptionalAttr<Modelica_Causality>:$causality,
        UnitAttr:$isProtected,
        OptionalAttr<StrAttr>:$description
    );

    let regions = (region
        MaxSizedRegion<1>:$modificationRegion,
        MaxSizedRegion<1>:$dimensionsRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Type":$type)>
    ];

    let extraClassDeclaration = [{
        /// Get the number of dimensions of the enclosing instances.
        size_t getNumEnclosingDimensions();

        /// Get the value of the declaration equation, if any.
        ::mlir::Value getBinding();

        /// Get the operation modifying the given attribute, if any.
        ModifyOp getAttributeModification(::llvm::StringRef name);
    }];
}

#endif // MARCO_DIALECT_MODELICA_IR_OPS_TD
