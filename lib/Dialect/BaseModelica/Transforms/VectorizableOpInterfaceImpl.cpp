#include "marco/Dialect/BaseModelica/Transforms/VectorizableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::mlir::bmodelica;

namespace {
struct CallOpInterface
    : public VectorizableOpInterface::ExternalModel<CallOpInterface, CallOp> {
  unsigned int
  getArgExpectedRank(mlir::Operation *op, unsigned int argIndex,
                     mlir::SymbolTableCollection &symbolTableCollection) const {
    auto callOp = mlir::cast<CallOp>(op);

    auto fallBackRankFn = [&]() -> unsigned int {
      mlir::Type argType = callOp.getArgs()[argIndex].getType();

      if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(argType)) {
        return shapedType.getRank();
      }

      return 0;
    };

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    auto calleeOp =
        resolveSymbol(moduleOp, symbolTableCollection, callOp.getCallee());

    if (calleeOp == nullptr) {
      // If the function is not declared, then assume that the arguments
      // types already match its hypothetical signature.
      return fallBackRankFn();
    }

    auto functionOp = mlir::dyn_cast<FunctionOp>(calleeOp);

    if (!functionOp) {
      return fallBackRankFn();
    }

    mlir::Type argType = functionOp.getArgumentTypes()[argIndex];

    if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(argType)) {
      return shapedType.getRank();
    }

    return 0;
  }

  mlir::LogicalResult
  scalarize(mlir::Operation *op, mlir::OpBuilder &builder,
            mlir::ValueRange args, mlir::TypeRange resultTypes,
            llvm::SmallVectorImpl<mlir::Value> &results) const {
    auto callOp = mlir::cast<CallOp>(op);

    auto newCallOp = builder.create<CallOp>(callOp.getLoc(), callOp.getCallee(),
                                            resultTypes, args);

    for (mlir::Value result : newCallOp.getResults()) {
      results.push_back(result);
    }

    return mlir::success();
  }
};

template <typename Op>
struct SingleOperandAndResultOpInterface
    : public VectorizableOpInterface::ExternalModel<
          SingleOperandAndResultOpInterface<Op>, Op> {
  mlir::LogicalResult
  scalarize(mlir::Operation *op, mlir::OpBuilder &builder,
            mlir::ValueRange args, mlir::TypeRange resultTypes,
            llvm::SmallVectorImpl<mlir::Value> &results) const {
    auto scalarizedOp =
        builder.create<Op>(op->getLoc(), resultTypes[0], args[0]);

    results.push_back(scalarizedOp.getResult());
    return mlir::success();
  }
};

struct Atan2OpInterface
    : public VectorizableOpInterface::ExternalModel<Atan2OpInterface, Atan2Op> {
  mlir::LogicalResult
  scalarize(mlir::Operation *op, mlir::OpBuilder &builder,
            mlir::ValueRange args, mlir::TypeRange resultTypes,
            llvm::SmallVectorImpl<mlir::Value> &results) const {
    auto scalarizedOp =
        builder.create<Atan2Op>(op->getLoc(), resultTypes[0], args[0], args[1]);

    results.push_back(scalarizedOp.getResult());
    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
void registerVectorizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *context, BaseModelicaDialect *dialect) {
        CallOp::attachInterface<::CallOpInterface>(*context);
      });
}
} // namespace mlir::bmodelica
