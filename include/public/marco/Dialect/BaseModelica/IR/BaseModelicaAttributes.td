#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAATTRIBUTES_TD
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAATTRIBUTES_TD

include "marco/Dialect/BaseModelica/IR/BaseModelica.td"
include "marco/Dialect/BaseModelica/IR/BaseModelicaAttrInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Base Modelica attribute definitions
//===---------------------------------------------------------------------===//

class BaseModelica_Attr<string name,
                    list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<BaseModelica_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

def BaseModelica_BooleanAttr
    : BaseModelica_Attr<"Boolean", [TypedAttrInterface]>
{
    let summary = "Modelica Boolean attribute";

    let description = [{
        Attribute representing a Boolean-typed value.
    }];

    let mnemonic = "bool";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "bool":$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "bool":$value), [{
            return $_get(type.getContext(), type, value);
        }]>,
        AttrBuilderWithInferredContext<(ins "MLIRContext*":$context, "bool":$value), [{
            return $_get(context, BooleanType::get(context), value);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_IntegerAttr
    : BaseModelica_Attr<"Integer", [TypedAttrInterface]>
{
    let summary = "Modelica Integer attribute";

    let description = [{
        Attribute representing an Integer-typed value.
    }];

    let mnemonic = "int";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        APIntParameter<"Value">:$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "long":$value), [{
            return $_get(type.getContext(), type, llvm::APInt(sizeof(long) * 8, value, true));
        }]>,
        AttrBuilderWithInferredContext<(ins "MLIRContext*":$context, "long":$value), [{
            return $_get(context, IntegerType::get(context), llvm::APInt(sizeof(long) * 8, value, true));
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_RealAttr
    : BaseModelica_Attr<"Real", [TypedAttrInterface]>
{
    let summary = "Modelica Real attribute";

    let description = [{
        Attribute representing a Real-typed value.
    }];

    let mnemonic = "real";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        APFloatParameter<"">:$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "::mlir::Type":$type, "double":$value), [{
            return $_get(type.getContext(), type, ::llvm::APFloat(value));
        }]>,
        AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext*":$context, "double":$value), [{
            return $_get(context, RealType::get(context), ::llvm::APFloat(value));
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_DenseBooleanElementsAttr
    : BaseModelica_Attr<"DenseBooleanElements", [TypedAttrInterface]>
{
    let summary = "Modelica dense Bool elements attribute";

    let description = [{
        Attribute representing a Boolean-typed array value.
    }];

    let mnemonic = "dense_bool";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        ArrayRefParameter<"bool", "elements">:$values);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "llvm::ArrayRef<bool>":$values), [{
            return $_get(type.getContext(), type, values);
        }]>
    ];
}

def BaseModelica_DenseIntegerElementsAttr
    : BaseModelica_Attr<"DenseIntegerElements", [TypedAttrInterface]>
{
    let summary = "Modelica dense Integer elements attribute";

    let description = [{
        Attribute representing a Integer-typed array value.
    }];

    let mnemonic = "dense_int";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        ArrayRefParameter<"int64_t", "elements">:$values);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "llvm::ArrayRef<int64_t>":$values), [{
            return $_get(type.getContext(), type, values);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_DenseRealElementsAttr
    : BaseModelica_Attr<"DenseRealElements", [TypedAttrInterface]>
{
    let summary = "Modelica dense Real elements attribute";

    let description = [{
        Attribute representing a Real-typed array value.
    }];

    let mnemonic = "dense_real";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        ArrayRefParameter<"::llvm::APFloat", "elements">:$values);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "llvm::ArrayRef<double>":$values), [{
            llvm::SmallVector<llvm::APFloat> data;

            for (double value : values) {
                data.emplace_back(value);
            }

            return $_get(type.getContext(), type, data);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_InverseFunctionsAttr : BaseModelica_Attr<"InverseFunctions">
{
    let summary = "TODO";

    let description = [{
        TODO.
    }];

    let mnemonic = "inverse";
    let parameters = (ins SelfAllocationParameter<"InverseFunctionsMap", "Inverse functions map">:$inverseFunctionsMap);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        public:
            bool isInvertible(unsigned int argumentIndex) const;
            StringRef getFunction(unsigned int argumentIndex) const;
            ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;
    }];
}

def BaseModelica_FunctionDerivativeAttr
    : BaseModelica_Attr<"FunctionDerivative">
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let mnemonic = "func_der";

    let parameters = (ins
        StringRefParameter<"Derived function name">:$name,
        "uint64_t":$order);

    let assemblyFormat = "$name `,` $order";
}

def BaseModelica_EquationPathAttr : BaseModelica_Attr<"EquationPath">
{
    let summary = "Equation path.";

    let description = [{
        Attribute representing a path within the IR of an equation.
    }];

    let mnemonic = "equation_path";
    let parameters = (ins "EquationPath":$value);
    let assemblyFormat = "$value";
}

def BaseModelica_IntegerRangeAttr : BaseModelica_Attr<"IntegerRange",
    [TypedAttrInterface,
    DeclareAttrInterfaceMethods<RangeAttrInterface>]>
{
    let summary = "Integer range attribute";

    let description = [{
        Attribute representing a range of Integer values.
    }];

    let mnemonic = "int_range";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "int64_t":$lowerBound,
        "int64_t":$upperBound,
        "int64_t":$step);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        AttrBuilder<(ins "int64_t":$lowerBound,
                         "int64_t":$upperBound,
                         "int64_t":$step)>
    ];
}

def BaseModelica_RealRangeAttr : BaseModelica_Attr<"RealRange",
    [TypedAttrInterface,
    DeclareAttrInterfaceMethods<RangeAttrInterface>]>
{
    let summary = "Real range attribute";

    let description = [{
        Attribute representing a range of Real values.
    }];

    let mnemonic = "real_range";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        APFloatParameter<"Lower bound">:$lowerBound,
        APFloatParameter<"Upper bound">:$upperBound,
        APFloatParameter<"Step">:$step);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        AttrBuilderWithInferredContext<(ins "mlir::Type":$type,
                         "double":$lowerBound,
                         "double":$upperBound,
                         "double":$step)>,
        AttrBuilder<(ins "double":$lowerBound,
                         "double":$upperBound,
                         "double":$step)>
    ];
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAATTRIBUTES_TD
