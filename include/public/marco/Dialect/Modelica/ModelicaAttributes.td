#ifndef MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica attribute definitions
//===----------------------------------------------------------------------===//

class Modelica_Attr<string name, string attrMnemonic>
    : AttrDef<Modelica_Dialect, name> {
    let mnemonic = attrMnemonic;
}

def Modelica_BooleanAttr : Modelica_Attr<"Boolean", "bool"> {
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

    let typeBuilder = "$_type";
}

def Modelica_IntegerAttr : Modelica_Attr<"Integer", "int"> {
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

    let typeBuilder = "$_type";
}

def Modelica_RealAttr : Modelica_Attr<"Real", "real"> {
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

    let typeBuilder = "$_type";
}

def Modelica_InverseFunctionsAttr : Modelica_Attr<"InverseFunctions", "derivative"> {
    let summary = "Modelica's Real attribute";

    let description = [{
        Type representing Modelica's Real.
    }];

    let parameters = (ins SelfAllocationParameter<"InverseFunctionsMap", "Inverse functions map">:$inverseFunctionsMap);

    let extraClassDeclaration = [{
        public:
            bool isInvertible(unsigned int argumentIndex) const;
            StringRef getFunction(unsigned int argumentIndex) const;
            ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;
    }];
}

def Modelica_DerivativeAttr : Modelica_Attr<"Derivative", "derivative"> {
    let summary = "Modelica's Real attribute";

    let description = [{
        Type representing Modelica's Real.
    }];

    let parameters = (ins StringRefParameter<"Derived function name">:$name, "unsigned int":$order);
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
