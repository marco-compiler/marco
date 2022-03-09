#ifndef MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
#define MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD

include "marco/dialects/modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica interface definitions
//===----------------------------------------------------------------------===//

class Modelica_Interface<string name> : OpInterface<name> {
    let cppNamespace = "::mlir::modelica";
}

def ClassInterface : Modelica_Interface<"ClassInterface"> {
    let description = [{
        Interface for a Modelica class.
    }];

    let methods = [
        InterfaceMethod<
            "Get the members of the class together with their names.",
            "unsigned int", "getArgExpectedRank",
            (ins "llvm::SmallVectorImpl<mlir::Value>&":$members, "llvm::SmallVectorImpl<llvm::StringRef>&":$names)
        >
    ];
}

def VectorizableOpInterface : Modelica_Interface<"VectorizableOpInterface"> {
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
            (ins "unsigned int":$argIndex)
        >,
        InterfaceMethod<
            "Convert the vectorized operation into a scalar one.",
            "mlir::ValueRange", "scalarize",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::ValueRange":$indexes)
        >,
    ];
}

def InvertibleOpInterface : Modelica_Interface<"InvertibleOpInterface"> {
    let description = [{
        This interface is used to abstract an operation that can be propagated
        down into the operations tree (i.e. towards the leaf values).
    }];

    let methods = [
        InterfaceMethod<
            "Invert the operation with respect to one of its arguments.",
            "::mlir::LogicalResult", "invert",
            (ins "::mlir::OpBuilder&":$builder, "unsigned int":$argumentIndex, "::mlir::ValueRange":$currentResult)
        >
    ];
}

def DistributableOpInterface : Modelica_Interface<"DistributableOpInterface"> {
    let description = [{
        Interface for an distributable operation.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute the operation among its arguments, if possible. The returned value is the one that will replace the previous operation result",
            "::mlir::Value", "distribute",
            (ins "::mlir::OpBuilder&":$builder)
        >
    ];
}

def NegateOpDistributionInterface : Modelica_Interface<"NegateOpDistributionInterface"> {
    let description = [{
        Interface to describe an operation that can accept the NegateOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a negate operation among the arguments. The returned value is the result after propagation",
            "::mlir::Value", "distributeNegateOp",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::Type":$resultType)
        >
    ];
}

def MulOpDistributionInterface : Modelica_Interface<"MulOpDistributionInterface"> {
    let description = [{
        Interface to describe an operation that can accept the MulOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a multiplication operation among the arguments. The returned value is the result after propagation",
            "::mlir::Value", "distributeMulOp",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::Type":$resultType, "::mlir::Value":$factor)
        >
    ];
}

def DivOpDistributionInterface : Modelica_Interface<"DivOpDistributionInterface"> {
    let description = [{
        Interface to describe an operation that can accept the DivOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a division operation among the arguments. The returned value is the result after propagation",
            "::mlir::Value", "distributeDivOp",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::Type":$resultType, "::mlir::Value":$divisor)
        >
    ];
}

def DerivableOpInterface : Modelica_Interface<"DerivableOpInterface"> {
    let description = [{
        Interface to describe an operation that can accept the DivOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Compute the derivative of the operation.",
            "mlir::ValueRange", "derive",
            (ins "::mlir::OpBuilder&":$builder, "::mlir::BlockAndValueMapping&":$derivatives)
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

def HeapAllocator : Modelica_Interface<"HeapAllocator"> {
    let description = [{
        Interface to describe an operation that can allocate memory on the heap.
    }];

    let methods = [
        InterfaceMethod<
            "Get whether the result of the operation should be deallocated.",
            "bool", "shouldBeDeallocated", (ins), [{
                return $_op->template getAttrOfType<::mlir::BoolAttr>("auto_dealloc").getValue();
            }]
        >,
        InterfaceMethod<
            "Set the value as automatically deallocated.",
            "void", "setAsAutomaticallyDeallocated", (ins), [{
                auto attr = ::mlir::BoolAttr::get($_op->getContext(), true);
                $_op->setAttr("auto_dealloc", attr);
            }]
        >,
        InterfaceMethod<
            "Set the value as manually deallocated.",
            "void", "setAsManuallyDeallocated", (ins), [{
                auto attr = ::mlir::BoolAttr::get($_op->getContext(), false);
                $_op->setAttr("auto_dealloc", attr);
            }]
        >
    ];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICAINTERFACES_TD
