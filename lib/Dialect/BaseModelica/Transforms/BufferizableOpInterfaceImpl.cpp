#include "marco/Dialect/BaseModelica/Transforms/BufferizableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace ::mlir::bmodelica;
using namespace ::mlir::bufferization;

namespace {
struct RawVariableOpInterface
    : public BufferizableOpInterface::ExternalModel<RawVariableOpInterface,
                                                    RawVariableOp> {
  bool bufferizesToAllocation(mlir::Operation *op, mlir::Value value) const {
    return true;
  }

  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  mlir::LogicalResult bufferize(mlir::Operation *op,
                                mlir::RewriterBase &rewriter,
                                const BufferizationOptions &options,
                                BufferizationState &state) const {
    auto rawVariableOp = mlir::cast<RawVariableOp>(op);

    // Fold away the op if it has no uses.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> resultType =
        mlir::bufferization::getBufferType(rawVariableOp.getResult(), options,
                                           state);

    if (mlir::failed(resultType)) {
      return mlir::failure();
    }

    replaceOpWithNewBufferizedOp<RawVariableOp>(
        rewriter, op, *resultType, rawVariableOp.getName(),
        rawVariableOp.getDimensionsConstraints(),
        rawVariableOp.getDynamicSizes(), rawVariableOp.isOutput());

    return mlir::success();
  }

  mlir::FailureOr<mlir::BaseMemRefType>
  getBufferType(mlir::Operation *op, mlir::Value value,
                const BufferizationOptions &options,
                const BufferizationState &state,
                llvm::SmallVector<mlir::Value> &invocationStack) const {
    auto rawVariableOp = mlir::cast<RawVariableOp>(op);

    auto tensorType =
        mlir::cast<mlir::TensorType>(rawVariableOp.getVariable().getType());

    auto memRefType = mlir::MemRefType::get(tensorType.getShape(),
                                            tensorType.getElementType());

    return mlir::cast<mlir::BaseMemRefType>(memRefType);
  }
};

struct RawVariableGetOpInterface
    : public BufferizableOpInterface::ExternalModel<RawVariableGetOpInterface,
                                                    RawVariableGetOp> {
  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  mlir::LogicalResult bufferize(mlir::Operation *op,
                                mlir::RewriterBase &rewriter,
                                const BufferizationOptions &options,
                                BufferizationState &state) const {
    auto rawVariableGetOp = mlir::cast<RawVariableGetOp>(op);

    mlir::FailureOr<mlir::Value> memRef =
        getBuffer(rewriter, rawVariableGetOp.getVariable(), options, state);

    if (mlir::failed(memRef)) {
      return mlir::failure();
    }

    auto rawVariableOp = memRef->getDefiningOp<RawVariableOp>();

    if (!rawVariableOp) {
      return mlir::failure();
    }

    replaceOpWithNewBufferizedOp<RawVariableGetOp>(rewriter, op, rawVariableOp);

    return mlir::success();
  }

  AliasingValueList getAliasingValues(mlir::Operation *op,
                                      mlir::OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto rawVariableGetOp = mlir::cast<RawVariableGetOp>(op);

    if (rawVariableGetOp.isStaticArrayVariable()) {
      return {{rawVariableGetOp.getResult(), BufferRelation::Equivalent}};
    }

    return {};
  }
};

struct RawVariableSetOpInterface
    : public BufferizableOpInterface::ExternalModel<RawVariableSetOpInterface,
                                                    RawVariableSetOp> {
  bool bufferizesToAllocation(mlir::Operation *op, mlir::Value value) const {
    return false;
  }

  bool bufferizesToMemoryRead(mlir::Operation *op, mlir::OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto rawVariableSetOp = mlir::cast<RawVariableSetOp>(op);
    return opOperand == rawVariableSetOp.getValueMutable();
  }

  bool bufferizesToMemoryWrite(mlir::Operation *op, mlir::OpOperand &opOperand,
                               const AnalysisState &state) const {
    auto rawVariableSetOp = mlir::cast<RawVariableSetOp>(op);
    return opOperand == rawVariableSetOp.getVariableMutable();
  }

  mlir::LogicalResult bufferize(mlir::Operation *op,
                                mlir::RewriterBase &rewriter,
                                const BufferizationOptions &options,
                                BufferizationState &state) const {
    auto rawVariableSetOp = mlir::cast<RawVariableSetOp>(op);

    mlir::FailureOr<mlir::Value> variableBuffer =
        getBuffer(rewriter, rawVariableSetOp.getVariable(), options, state);

    if (mlir::failed(variableBuffer)) {
      return mlir::failure();
    }

    auto rawVariableOp = variableBuffer->getDefiningOp<RawVariableOp>();

    if (!rawVariableOp) {
      return mlir::failure();
    }

    if (rawVariableSetOp.isScalarVariable()) {
      replaceOpWithNewBufferizedOp<RawVariableSetOp>(
          rewriter, op, rawVariableOp, rawVariableSetOp.getValue());
    } else {
      mlir::FailureOr<mlir::Value> valueBuffer =
          getBuffer(rewriter, rawVariableSetOp.getValue(), options, state);

      if (mlir::failed(valueBuffer)) {
        return mlir::failure();
      }

      replaceOpWithNewBufferizedOp<RawVariableSetOp>(
          rewriter, op, rawVariableOp, *valueBuffer);
    }

    return mlir::success();
  }

  AliasingValueList getAliasingValues(mlir::Operation *op,
                                      mlir::OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto rawVariableSetOp = mlir::cast<RawVariableSetOp>(op);

    if (opOperand == rawVariableSetOp.getVariableMutable()) {
      return {{rawVariableSetOp.getVariable(), BufferRelation::Equivalent}};
    }

    return {};
  }
};
} // namespace

namespace mlir::bmodelica {
void registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            BaseModelicaDialect *dialect) {
    RawVariableOp::attachInterface<::RawVariableOpInterface>(*context);
    RawVariableGetOp::attachInterface<::RawVariableGetOpInterface>(*context);
    RawVariableSetOp::attachInterface<::RawVariableSetOpInterface>(*context);
  });
}
} // namespace mlir::bmodelica
