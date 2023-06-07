#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica attribute definitions
//===---------------------------------------------------------------------===//

class Modelica_Attr<string name,
                    list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<Modelica_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

def Modelica_BooleanAttr
    : Modelica_Attr<"Boolean", [TypedAttrInterface]>
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

def Modelica_IntegerAttr
    : Modelica_Attr<"Integer", [TypedAttrInterface]>
{
    let summary = "Modelica Integer attribute";

    let description = [{
        Attribute representing an Integer-typed value.
    }];

    let mnemonic = "int";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "::llvm::APInt":$value);

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

def Modelica_RealAttr
    : Modelica_Attr<"Real", [TypedAttrInterface]>
{
    let summary = "Modelica Real attribute";

    let description = [{
        Attribute representing a Real-typed value.
    }];

    let mnemonic = "real";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "::llvm::APFloat":$value);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "double":$value), [{
            return $_get(type.getContext(), type, ::llvm::APFloat(value));
        }]>,
        AttrBuilderWithInferredContext<(ins "MLIRContext*":$context, "double":$value), [{
            return $_get(context, RealType::get(context), ::llvm::APFloat(value));
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def Modelica_BooleanArrayAttr
    : Modelica_Attr<"BooleanArray", [TypedAttrInterface]>
{
    let summary = "Modelica Bool array attribute";

    let description = [{
        Attribute representing a Boolean-typed array value.
    }];

    let mnemonic = "bool_array";

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

def Modelica_IntegerArrayAttr
    : Modelica_Attr<"IntegerArray", [TypedAttrInterface]>
{
    let summary = "Modelica Integer array attribute";

    let description = [{
        Attribute representing a Integer-typed array value.
    }];

    let mnemonic = "integer_array";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        ArrayRefParameter<"::llvm::APInt", "elements">:$values);

    let builders = [
        AttrBuilderWithInferredContext<(ins "Type":$type, "llvm::ArrayRef<int64_t>":$values), [{
            llvm::SmallVector<llvm::APInt> data;

            for (int64_t value : values) {
                data.push_back(llvm::APInt(64, value, true));
            }

            return $_get(type.getContext(), type, data);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
}

def Modelica_RealArrayAttr
    : Modelica_Attr<"RealArray", [TypedAttrInterface]>
{
    let summary = "Modelica Real array attribute";

    let description = [{
        Attribute representing a Real-typed array value.
    }];

    let mnemonic = "real_array";

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

def Modelica_InverseFunctionsAttr
    : Modelica_Attr<"InverseFunctions"> {
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

def Modelica_DerivativeAttr
    : Modelica_Attr<"Derivative"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let mnemonic = "derivative";
    let parameters = (ins StringRefParameter<"Derived function name">:$name, "unsigned int":$order);

    let hasCustomAssemblyFormat = 1;
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
