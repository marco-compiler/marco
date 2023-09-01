#ifndef MARCO_DIALECTS_SIMULATION_SIMULATIONATTRIBUTES_TD
#define MARCO_DIALECTS_SIMULATION_SIMULATIONATTRIBUTES_TD

include "marco/Dialect/Simulation/SimulationDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Simulation attribute definitions
//===---------------------------------------------------------------------===//

class Simulation_Attr<
    string name,
    string attrMnemonic,
    list<Trait> traits = []>
    : AttrDef<Simulation_Dialect, name, traits>
{
    let mnemonic = attrMnemonic;
}

def ArrayOfDimensionsParam
    : OptionalArrayRefParameter<"int64_t">
{
    let printer = [{
        $_printer << "[";
        $_printer.printStrippedAttrOrType($_self);
        $_printer << "]";
    }];

    let parser = [{
        [&]() -> FailureOr<::llvm::SmallVector<int64_t>> {
            using ResultType = ::llvm::SmallVector<int64_t>;
            if ($_parser.parseLSquare()) {
                return ::mlir::failure();
            }

            if ($_parser.parseOptionalRSquare().failed()) {
                auto values = ::mlir::FieldParser<ResultType>::parse($_parser);

                if (mlir::failed(values) ||
                    $_parser.parseRSquare()) {
                    return mlir::failure();
                }

                return *values;
            }

            return ResultType();
        }() }];
}

def ArrayOfMultidimensionalRangeParam
    : OptionalArrayRefParameter<"MultidimensionalRangeAttr">
{
    let printer = [{
        $_printer << "[";
        $_printer.printStrippedAttrOrType($_self);
        $_printer << "]";
    }];

    let parser = [{
        [&]() -> FailureOr<::llvm::SmallVector<MultidimensionalRangeAttr>> {
            using ResultType = ::llvm::SmallVector<MultidimensionalRangeAttr>;
            if ($_parser.parseLSquare()) {
                return ::mlir::failure();
            }

            if ($_parser.parseOptionalRSquare().failed()) {
                auto values = ::mlir::FieldParser<ResultType>::parse($_parser);

                if (mlir::failed(values) ||
                    $_parser.parseRSquare()) {
                    return mlir::failure();
                }

                return *values;
            }

            return ResultType();
        }() }];
}

def Simulation_VariableAttr
    : Simulation_Attr<"Variable", "variable", [TypedAttrInterface]>
{
    let summary = "Variable.";

    let description = [{
        Attribute representing a variable.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"Type of the variable", "mlir::Type", "$_builder.getF64Type()">:$type,
        StringRefParameter<"Name">:$name,
        ArrayOfDimensionsParam:$dimensions,
        DefaultValuedParameter<"bool", "false">:$printable,
        ArrayOfMultidimensionalRangeParam:$printable_indices);

    let assemblyFormat = "`<` struct(params) `>`";

    let extraClassDeclaration = [{
        int64_t getRank() const;
    }];
}

def VariableAttr
    : Attr<CPred<"$_self.isa<::mlir::simulation::VariableAttr>()">,
           "Variable attribute">
{
    let storageType = [{ ::mlir::simulation::VariableAttr }];
    let returnType = [{ ::llvm::StringRef }];
    let convertFromStorage = [{ $_self.getName() }];
    let valueType = AnyType;
}

def VariablesListAttr
    : TypedArrayAttrBase<VariableAttr, "Variable array attribute">;

def Simulation_DerivativeAttr
    : Simulation_Attr<"Derivative", "derivative", []>
{
    let summary = "Derivative.";

    let description = [{
        Attribute representing a map between a variable and its derivative.
    }];

    let parameters = (ins "VariableAttr":$variable, "VariableAttr":$derivative);

    let assemblyFormat = "`<` $variable `->` $derivative `>`";
}

def DerivativeAttr
    : Attr<CPred<"$_self.isa<::mlir::simulation::DerivativeAttr>()">,
           "Derivative attribute">;

def DerivativesListAttr
    : TypedArrayAttrBase<DerivativeAttr, "Derivative array attribute">;

#endif // MARCO_DIALECTS_SIMULATION_SIMULATIONATTRIBUTES_TD
