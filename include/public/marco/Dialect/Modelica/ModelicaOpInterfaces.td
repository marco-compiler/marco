#ifndef MARCO_DIALECTS_MODELICA_MODELICAOPINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAOPINTERFACES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Modelica op interface definitions
//===---------------------------------------------------------------------===//

class Modelica_OpInterface<string name>
    : OpInterface<name>
{
    let cppNamespace = "::mlir::modelica";
}

def ClassInterface : Modelica_OpInterface<"ClassInterface">
{
    let description = [{
        Interface for a Modelica class.
    }];
}

def VectorizableOpInterface : Modelica_OpInterface<"VectorizableOpInterface">
{
    let description = [{
        Interface for a vectorizable operation.
    }];

    let methods = [
        InterfaceMethod<
            "Get the arguments of the operation.",
            "::mlir::ValueRange", "getArgs"
        >,
        InterfaceMethod<
            "Get the expected rank of an argument in case of a scalar usage of the operation.",
            "unsigned int", "getArgExpectedRank",
            (ins "unsigned int":$argIndex, "::mlir::SymbolTableCollection&":$symbolTable)
        >,
        InterfaceMethod<
            "Convert the vectorized operation into a scalar one.",
            "mlir::ValueRange", "scalarize",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::ValueRange":$indexes)
        >,
    ];
}

def EquationInterface : Modelica_OpInterface<"EquationInterface">
{
}

def EquationExpressionOpInterface
    : Modelica_OpInterface<"EquationExpressionOpInterface">
{
    let description = [{
        This interface is used to represent an operation that can be used to
        represent an expression within an equation.
    }];

    let methods = [
        InterfaceMethod<
            "Print inline representation of the expression.",
            "void", "printExpression",
            (ins "::llvm::raw_ostream&":$os,
                 "const ::llvm::DenseMap<::mlir::Value, int64_t>&":$inductions)>,
        InterfaceMethod<
            "Get the number of elements.",
            "uint64_t", "getNumOfExpressionElements",
            (ins), [{}], [{
                return $_op.getOperation()->getNumOperands();
            }]>,
        InterfaceMethod<
            "Get the element at a given position.",
            "::mlir::Value", "getExpressionElement",
            (ins "uint64_t":$position), [{}], [{
                return $_op.getOperation()->getOperand(position);
            }]>,
        InterfaceMethod<
            "Get the element at a given position.",
            "::mlir::LogicalResult", "mapAdditionalInductions",
            (ins "AdditionalInductions&":$additionalInductions),
            [{}], [{
                return mlir::success();
            }]>,
        InterfaceMethod<
            "Get the accesses of this expression",
            "::mlir::LogicalResult", "getEquationAccesses",
            (ins "::llvm::SmallVectorImpl<VariableAccess>&":$accesses,
                 "::mlir::SymbolTableCollection&":$symbolTable,
                 "::llvm::DenseMap<::mlir::Value, unsigned int>&":$explicitInductionsPositionMap,
                 "AdditionalInductions&":$additionalInductions,
                 "::llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>&":$dimensionAccesses,
                 "EquationPath":$path), [{}], [{
                auto opInterface = mlir::cast<EquationExpressionOpInterface>($_op.getOperation());
                uint64_t numOfChildren = opInterface.getNumOfExpressionElements();

                if (mlir::failed(opInterface.mapAdditionalInductions(additionalInductions))) {
                    return mlir::failure();
                }

                llvm::SmallVector<std::unique_ptr<DimensionAccess>> childDimensionAccesses;

                for (uint64_t i = 0; i < numOfChildren; ++i) {
                    childDimensionAccesses.clear();
                    mlir::Value child = opInterface.getExpressionElement(i);
                    auto childOp = child.getDefiningOp();

                    if (!childOp) {
                        continue;
                    }

                    auto childExpressionInt =
                        mlir::dyn_cast<EquationExpressionOpInterface>(childOp);

                    if (!childExpressionInt) {
                        return mlir::failure();
                    }

                    if (mlir::failed(childExpressionInt.getEquationAccesses(
                            accesses, symbolTable,
                            explicitInductionsPositionMap,
                            additionalInductions,
                            childDimensionAccesses,
                            path + i))) {
                       return mlir::failure();
                    }
                }

                return mlir::success();
             }]>
    ];
}

def InvertibleOpInterface : Modelica_OpInterface<"InvertibleOpInterface">
{
    let description = [{
        This interface is used to abstract an operation that can be propagated
        down into the operations tree (i.e. towards the leaf values).
    }];

    let methods = [
        InterfaceMethod<
            "Invert the operation with respect to one of its arguments.",
            "::mlir::Value", "inverse",
            (ins "::mlir::OpBuilder&":$builder,
                 "unsigned int":$argumentIndex,
                 "::mlir::ValueRange":$currentResult)
        >
    ];
}

def DistributableOpInterface : Modelica_OpInterface<"DistributableOpInterface">
{
    let description = [{
        Interface for an distributable operation.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute the operation among its arguments, if possible. The returned value is the one that will replace the previous operation result",
            "::mlir::LogicalResult", "distribute",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder)
        >
    ];
}

def NegateOpDistributionInterface : Modelica_OpInterface<"NegateOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the NegateOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a negate operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeNegateOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder,
                 "::mlir::Type":$resultType)
        >
    ];
}

def MulOpDistributionInterface : Modelica_OpInterface<"MulOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the MulOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a multiplication operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeMulOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder,
                 "::mlir::Type":$resultType,
                 "::mlir::Value":$factor)
        >
    ];
}

def DivOpDistributionInterface : Modelica_OpInterface<"DivOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the DivOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a division operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeDivOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder,
                 "::mlir::Type":$resultType,
                 "::mlir::Value":$divisor)
        >
    ];
}

def DerivableOpInterface : Modelica_OpInterface<"DerivableOpInterface">
{
    let description = [{
        Interface to describe an operation that can be traversed while
        performing the automatic differentiation.
    }];

    let methods = [
        InterfaceMethod<
            "Compute the derivative of the operation.",
            "mlir::ValueRange", "derive",
            (ins
                "::mlir::OpBuilder&":$builder,
                "const ::llvm::DenseMap<mlir::StringAttr, mlir::StringAttr>&":$symbolDerivatives,
                "::mlir::IRMapping&":$ssaDerivatives)
        >,
        InterfaceMethod<
            "Get the operands to be derived.",
            "void", "getOperandsToBeDerived",
            (ins "::llvm::SmallVectorImpl<mlir::Value>&":$toBeDerived)
        >,
        InterfaceMethod<
            "Get the regions to be derived.",
            "void", "getDerivableRegions",
            (ins "::llvm::SmallVectorImpl<mlir::Region*>&":$regions)
        >
    ];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAOPINTERFACES_TD
