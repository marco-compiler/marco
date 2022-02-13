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
// AbsOp
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

//===----------------------------------------------------------------------===//
// AcosOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AddOp
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
        and in this case the result must have the same shape.
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// AddEWOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def Modelica_AndOp : Modelica_Op<"and",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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
// ArrayCastOp
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

//===----------------------------------------------------------------------===//
// ArrayCloneOp
//===----------------------------------------------------------------------===//

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
}

//===----------------------------------------------------------------------===//
// AsinOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// AtanOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Atan2Op
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// BreakOp
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

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// CoshOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// DerOp
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
// DiagonalOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// DivEWOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// EqOp
//===----------------------------------------------------------------------===//

def Modelica_EqOp : Modelica_Op<"eq", [Commutative]>
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
// ExpOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// FreeOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// GtOp
//===----------------------------------------------------------------------===//

def Modelica_GtOp : Modelica_Op<"gt", []>
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
// GteOp
//===----------------------------------------------------------------------===//

def Modelica_GteOp : Modelica_Op<"gte", []>
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
// IdentityOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// LinspaceOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// LogOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Log10Op
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// LtOp
//===----------------------------------------------------------------------===//

def Modelica_LtOp : Modelica_Op<"lt", []>
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
// LteOp
//===----------------------------------------------------------------------===//

def Modelica_LteOp : Modelica_Op<"lte", []>
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
// NDimsOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// MulEWOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// NegateOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

def Modelica_NotOp : Modelica_Op<"not",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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
// NotEqOp
//===----------------------------------------------------------------------===//

def Modelica_NotEqOp : Modelica_Op<"neq", [Commutative]>
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
// OnesOp
//===----------------------------------------------------------------------===//

def Modelica_OnesOp : Modelica_Op<"ones",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$sizes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` type($sizes) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def Modelica_OrOp : Modelica_Op<"or",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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
// PowOp
//===----------------------------------------------------------------------===//

def Modelica_PowOp : Modelica_Op<"pow",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
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
// PowEWOp
//===----------------------------------------------------------------------===//

def Modelica_PowEWOp : Modelica_Op<"pow_ew",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
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
// PrintOp
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

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SignOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SinhOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SizeOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SubEWOp
//===----------------------------------------------------------------------===//

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
// SubscriptionOp
//===----------------------------------------------------------------------===//

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
// SumOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SymmetricOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TanOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

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
// ZerosOp
//===----------------------------------------------------------------------===//

def Modelica_ZerosOp : Modelica_Op<"zeros",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins AnyType:$sizes);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` type($sizes) `->` type($result)
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICA_TD
