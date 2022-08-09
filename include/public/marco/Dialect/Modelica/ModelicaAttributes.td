#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Modelica attribute definitions
//===----------------------------------------------------------------------===//

class Modelica_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Modelica_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def Modelica_BooleanAttr : Modelica_Attr<"Boolean", "bool", [TypedAttrInterface]> {
    let summary = "Modelica's Boolean attribute";

    let description = [{
        Attribute representing a Boolean-typed value.
    }];

    let parameters = (ins "Type":$type, "bool":$value);

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

def Modelica_IntegerAttr : Modelica_Attr<"Integer", "int", [TypedAttrInterface]> {
    let summary = "Modelica's Integer attribute";

    let description = [{
        Attribute representing an Integer-typed value.
    }];

    let parameters = (ins "Type":$type, "::llvm::APInt":$value);

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

def Modelica_RealAttr : Modelica_Attr<"Real", "real", [TypedAttrInterface]> {
    let summary = "Modelica's Real attribute";

    let description = [{
        Attribute representing a Real-typed value.
    }];

    let parameters = (ins "Type":$type, "::llvm::APFloat":$value);

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

def Modelica_InverseFunctionsAttr : Modelica_Attr<"InverseFunctions", "derivative"> {
    let summary = "TODO";

    let description = [{
        TODO.
    }];

    let parameters = (ins SelfAllocationParameter<"InverseFunctionsMap", "Inverse functions map">:$inverseFunctionsMap);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        public:
            bool isInvertible(unsigned int argumentIndex) const;
            StringRef getFunction(unsigned int argumentIndex) const;
            ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;
    }];
}

def Modelica_DerivativeAttr : Modelica_Attr<"Derivative", "derivative"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let parameters = (ins StringRefParameter<"Derived function name">:$name, "unsigned int":$order);

    let hasCustomAssemblyFormat = 1;
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
