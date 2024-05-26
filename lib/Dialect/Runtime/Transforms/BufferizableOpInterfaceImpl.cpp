#include "marco/Dialect/Runtime/Transforms/BufferizableOpInterfaceImpl.h"
#include "marco/Dialect/Runtime/IR/RuntimeDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace ::mlir::runtime;
using namespace ::mlir::bufferization;

namespace
{
  struct FunctionOpInterface
      : public BufferizableOpInterface::ExternalModel<
            FunctionOpInterface, FunctionOp>
  {
    mlir::LogicalResult bufferize(
        mlir::Operation* op,
        mlir::RewriterBase& rewriter,
        const BufferizationOptions& options) const
    {
      auto functionOp = mlir::cast<FunctionOp>(op);

      llvm::SmallVector<mlir::Type> args;
      llvm::SmallVector<mlir::Type> results;

      for (mlir::Type arg : functionOp.getArgumentTypes()) {
        if (auto tensorType = arg.dyn_cast<mlir::RankedTensorType>()) {
          args.push_back(mlir::MemRefType::get(
              tensorType.getShape(), tensorType.getElementType()));

          continue;
        }

        if (auto tensorType = arg.dyn_cast<mlir::UnrankedTensorType>()) {
          auto memSpace = options.defaultMemorySpaceFn(tensorType);

          if (!memSpace) {
            return mlir::failure();
          }

          args.push_back(mlir::UnrankedMemRefType::get(
              tensorType.getElementType(), *memSpace));

          continue;
        }

        args.push_back(arg);
      }

      auto visibility = functionOp.getVisibility();

      auto replacement = replaceOpWithNewBufferizedOp<FunctionOp>(
          rewriter, op, functionOp.getSymName(),
          rewriter.getFunctionType(args, results));

      replacement.setVisibility(visibility);
      return mlir::success();
    }

    bool hasTensorSemantics(mlir::Operation* op) const
    {
      auto isaTensor = [](mlir::Type type) {
        return mlir::isa<
            mlir::RankedTensorType,
            mlir::UnrankedTensorType>(type);
      };

      auto functionOp = mlir::cast<FunctionOp>(op);

      bool hasTensorArg =
          llvm::any_of(functionOp.getArgumentTypes(), isaTensor);

      bool hasTensorResult =
          llvm::any_of(functionOp.getResultTypes(), isaTensor);

      if (hasTensorArg || hasTensorResult) {
        return true;
      }

      return false;
    }
  };

  struct CallOpInterface
      : public BufferizableOpInterface::ExternalModel<
            CallOpInterface, CallOp>
  {
    bool bufferizesToMemoryRead(
        mlir::Operation* op,
        mlir::OpOperand& opOperand,
        const AnalysisState& state) const
    {
      return false;
    }

    bool bufferizesToMemoryWrite(
        mlir::Operation* op,
        mlir::OpOperand& opOperand,
        const AnalysisState& state) const
    {
      return false;
    }

    mlir::LogicalResult bufferize(
        mlir::Operation* op,
        mlir::RewriterBase& rewriter,
        const BufferizationOptions& options) const
    {
      auto callOp = mlir::cast<CallOp>(op);
      llvm::SmallVector<mlir::Value> args;

      for (mlir::Value arg : callOp.getArgs()) {
        if (arg.getType().isa<mlir::TensorType>()) {
          auto argBuffer = getBuffer(rewriter, arg, options);

          if (mlir::failed(argBuffer)) {
            return mlir::failure();
          }

          args.push_back(*argBuffer);
        } else {
          args.push_back(arg);
        }
      }

      replaceOpWithNewBufferizedOp<CallOp>(
          rewriter, op, callOp.getResultTypes(), callOp.getCallee(), args);

      return mlir::success();
    }

    AliasingValueList getAliasingValues(
        mlir::Operation* op,
        mlir::OpOperand& opOperand,
        const AnalysisState& state) const
    {
      return {{op->getOperand(opOperand.getOperandNumber()),
               BufferRelation::Equivalent}};
    }
  };
}

namespace mlir::runtime
{
  void registerBufferizableOpInterfaceExternalModels(
      mlir::DialectRegistry& registry)
  {
    registry.addExtension(+[](mlir::MLIRContext* context,
                              RuntimeDialect* dialect) {
      FunctionOp::attachInterface<::FunctionOpInterface>(*context);
      CallOp::attachInterface<::CallOpInterface>(*context);
    });
  }
}
