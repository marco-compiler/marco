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

    let mnemonic = "int_array";

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

def Modelica_InverseFunctionsAttr : Modelica_Attr<"InverseFunctions">
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

def Modelica_VarDerivativeAttr : Modelica_Attr<"VarDerivative">
{
    let summary = "Variable's derivative information.";

    let description = [{
        Attribute representing the information about a derivative of a variable.
    }];

    let mnemonic = "var_derivative";

    let parameters = (ins
        "mlir::SymbolRefAttr":$variable,
        "mlir::SymbolRefAttr":$derivative,
        OptionalParameter<"IndexSetAttr">:$derived_indices);

    let assemblyFormat = [{
        $variable `,` $derivative (`,` $derived_indices^)?
    }];
}

def Modelica_EquationViewAttr : Modelica_Attr<"EquationView">
{
    let summary = "Equation view.";

    let description = [{
        Attribute representing a view over an equation.
    }];

    let mnemonic = "equation_view";

    let parameters = (ins
        "int64_t":$sideElementIndex,
        "mlir::SymbolRefAttr":$component);

    let assemblyFormat = [{
        $sideElementIndex `,` $component
    }];
}

def Modelica_EquationPathAttr : Modelica_Attr<"EquationPath">
{
    let summary = "Equation path.";

    let description = [{
        Attribute representing a path within the IR of an equation.
    }];

    let mnemonic = "equation_path";
    let parameters = (ins "EquationPath":$value);
    let assemblyFormat = "$value";
}

def Modelica_EquationScheduleDirectionAttr
    : Modelica_Attr<"EquationScheduleDirection">
{
    let summary = "Equation's schedule iteration direction.";

    let description = [{
        Attribute representing the runtime iteration direction for the indices
        of a scheduled equation.
    }];

    let mnemonic = "equation_schedule_direction";
    let parameters = (ins "EquationScheduleDirection":$value);
    let assemblyFormat = "$value";
}

def Modelica_EquationMatchAttr : Modelica_Attr<"EquationMatch">
{
    let summary = "Equation match information.";

    let description = [{
        Attribute representing the match information for an equation.
    }];

    let mnemonic = "equation_match";

    let parameters = (ins
        "mlir::modeling::IndexSetAttr":$equation_indices,
        "EquationPathAttr":$path,
        OptionalParameter<"mlir::modeling::IndexSetAttr">:$variable_extra_indices,
        OptionalParameter<"mlir::SymbolRefAttr">:$component);

    let assemblyFormat = [{
        $equation_indices `,` $path (`,` $variable_extra_indices^)? (`,` $component^)?
    }];
}

def Modelica_IntegerRangeAttr : Modelica_Attr<"IntegerRange", [TypedAttrInterface]>
{
    let summary = "Iterable Integer range attribute";

    let description = [{
        Attribute representing an iterable range of Integer values.
    }];

    let mnemonic = "int_range";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "int64_t":$lowerBound,
        "int64_t":$upperBound,
        "int64_t":$step);

    let assemblyFormat = [{
        $lowerBound `,` $upperBound `,` $step `:` $type
    }];

    let builders = [
        AttrBuilder<(
            ins "int64_t":$lowerBound, "int64_t":$upperBound, "int64_t":$step)>
    ];
}

def Modelica_RealRangeAttr : Modelica_Attr<"RealRange", [TypedAttrInterface]>
{
    let summary = "Iterable Real range attribute";

    let description = [{
        Attribute representing an iterable range of Real values.
    }];

    let mnemonic = "real_range";

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        APFloatParameter<"Lower bound">:$lowerBound,
        APFloatParameter<"Upper bound">:$upperBound,
        APFloatParameter<"Step">:$step);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        AttrBuilder<(
            ins "double":$lowerBound, "double":$upperBound, "double":$step)>
    ];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAATTRIBUTES_TD
