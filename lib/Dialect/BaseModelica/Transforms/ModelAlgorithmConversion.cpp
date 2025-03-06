#include "marco/Dialect/BaseModelica/Transforms/ModelAlgorithmConversion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_MODELALGORITHMCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class AlgorithmOpPattern : public mlir::OpRewritePattern<AlgorithmOp> {
public:
  AlgorithmOpPattern(mlir::MLIRContext *context,
                     mlir::SymbolTableCollection &symbolTable,
                     size_t &algorithmsCounter)
      : mlir::OpRewritePattern<AlgorithmOp>(context), symbolTable(&symbolTable),
        algorithmsCounter(&algorithmsCounter) {}

  mlir::LogicalResult
  matchAndRewrite(AlgorithmOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto modelOp = op->getParentOfType<ModelOp>();

    // Determine the read and written variables.
    llvm::DenseSet<VariableOp> readVariables;
    llvm::DenseSet<VariableOp> writtenVariables;

    op.walk([&](VariableGetOp getOp) {
      auto variableOp = symbolTable->lookupSymbolIn<VariableOp>(
          modelOp, getOp.getVariableAttr());

      readVariables.insert(variableOp);
    });

    op.walk([&](VariableSetOp setOp) {
      auto variableOp = symbolTable->lookupSymbolIn<VariableOp>(
          modelOp, setOp.getVariableAttr());

      writtenVariables.insert(variableOp);
    });

    // Determine the input and output variables of the function.
    // If a variable is read, but not written, then it will be an argument
    // of the function. All written variables are results of the function.
    llvm::SmallVector<VariableOp> inputVariables;
    llvm::SmallVector<VariableOp> outputVariables(writtenVariables.begin(),
                                                  writtenVariables.end());

    for (VariableOp readVariable : readVariables) {
      if (!writtenVariables.contains(readVariable)) {
        inputVariables.push_back(readVariable);
      }
    }

    // Obtain a unique name for the function to be created.
    std::string functionName = getFunctionName(modelOp, op);

    // Create the function.
    auto moduleOp = modelOp->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto functionOp = rewriter.create<FunctionOp>(loc, functionName);
    rewriter.createBlock(&functionOp.getBodyRegion());

    // Declare the variables.
    rewriter.setInsertionPointToStart(functionOp.getBody());
    mlir::IRMapping mapping;

    for (VariableOp variableOp : inputVariables) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          rewriter.clone(*variableOp.getOperation(), mapping));

      auto originalVariableType = variableOp.getVariableType();

      clonedVariableOp.setType(
          VariableType::get(originalVariableType.getShape(),
                            originalVariableType.getElementType(),
                            VariabilityProperty::none, IOProperty::input));
    }

    for (VariableOp variableOp : outputVariables) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          rewriter.clone(*variableOp.getOperation(), mapping));

      auto originalVariableType = variableOp.getVariableType();

      clonedVariableOp.setType(
          VariableType::get(originalVariableType.getShape(),
                            originalVariableType.getElementType(),
                            VariabilityProperty::none, IOProperty::output));
    }

    // Set the default value of the output variables.
    for (VariableOp variableOp : outputVariables) {
      StartOp startOp = getStartOp(modelOp, variableOp.getSymName());

      auto defaultOp =
          rewriter.create<DefaultOp>(startOp.getLoc(), variableOp.getSymName());

      rewriter.cloneRegionBefore(startOp.getBodyRegion(),
                                 defaultOp.getBodyRegion(),
                                 defaultOp.getBodyRegion().end());

      if (startOp.getEach()) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);

        auto yieldOp = mlir::cast<YieldOp>(
            defaultOp.getBodyRegion().back().getTerminator());

        assert(yieldOp.getValues().size() == 1);
        rewriter.setInsertionPoint(yieldOp);

        mlir::Value array = rewriter.create<TensorBroadcastOp>(
            yieldOp.getLoc(), variableOp.getVariableType().unwrap(),
            yieldOp.getValues()[0]);

        rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, array);
      }
    }

    // Create the algorithm inside the function and move the original body
    // into it.
    rewriter.setInsertionPointToEnd(functionOp.getBody());
    auto algorithmOp = rewriter.create<AlgorithmOp>(loc);

    rewriter.inlineRegionBefore(op.getBodyRegion(), algorithmOp.getBodyRegion(),
                                algorithmOp.getBodyRegion().end());

    // Create the equation templates.
    llvm::SmallVector<EquationTemplateOp> templateOps;

    for (size_t i = 0, e = outputVariables.size(); i < e; ++i) {
      VariableOp outputVariable = outputVariables[i];
      VariableType variableType = outputVariable.getVariableType();
      int64_t rank = variableType.getRank();

      if (auto parentOp = mlir::dyn_cast<DynamicOp>(op->getParentOp())) {
        rewriter.setInsertionPoint(parentOp);
      }

      if (auto parentOp = mlir::dyn_cast<InitialOp>(op->getParentOp())) {
        rewriter.setInsertionPoint(parentOp);
      }

      auto templateOp = rewriter.create<EquationTemplateOp>(loc);
      templateOps.push_back(templateOp);
      mlir::Block *templateBody = templateOp.createBody(rank);
      rewriter.setInsertionPointToStart(templateBody);

      llvm::SmallVector<mlir::Value> inputVariableGetOps;

      for (VariableOp variable : inputVariables) {
        inputVariableGetOps.push_back(
            rewriter.create<VariableGetOp>(loc, variable));
      }

      auto callOp =
          rewriter.create<CallOp>(loc, functionOp, inputVariableGetOps);

      mlir::Value lhs = rewriter.create<VariableGetOp>(loc, outputVariables[i]);

      mlir::Value rhs = callOp.getResult(i);

      if (auto inductions = templateOp.getInductionVariables();
          !inductions.empty()) {
        lhs = rewriter.create<TensorExtractOp>(
            lhs.getLoc(), lhs, templateOp.getInductionVariables());

        rhs = rewriter.create<TensorExtractOp>(
            rhs.getLoc(), rhs, templateOp.getInductionVariables());
      }

      mlir::Value lhsOp = rewriter.create<EquationSideOp>(loc, lhs);
      mlir::Value rhsOp = rewriter.create<EquationSideOp>(loc, rhs);
      rewriter.create<EquationSidesOp>(loc, lhsOp, rhsOp);
    }

    // Create the equation instances.
    rewriter.setInsertionPoint(op);

    for (size_t i = 0, e = outputVariables.size(); i < e; ++i) {
      EquationTemplateOp templateOp = templateOps[i];

      auto instanceOp =
          rewriter.create<EquationInstanceOp>(templateOp.getLoc(), templateOp);

      if (mlir::failed(instanceOp.setIndices(outputVariables[i].getIndices(),
                                             *symbolTable))) {
        return mlir::failure();
      }
    }

    // Erase the algorithm.
    rewriter.eraseOp(op);

    return mlir::success();
  }

private:
  std::string getFunctionName(ModelOp modelOp, AlgorithmOp algorithmOp) const {
    return modelOp.getSymName().str() + "_algorithm_" +
           std::to_string((*algorithmsCounter)++);
  }

  StartOp getStartOp(ModelOp modelOp, llvm::StringRef variable) const {
    for (StartOp startOp : modelOp.getOps<StartOp>()) {
      assert(startOp.getVariable().getNestedReferences().empty());

      if (startOp.getVariable().getRootReference() == variable) {
        return startOp;
      }
    }

    llvm_unreachable("StartOp not found");
    return nullptr;
  }

private:
  mlir::SymbolTableCollection *symbolTable;
  size_t *algorithmsCounter;
};
} // namespace

namespace {
class ModelAlgorithmConversionPass
    : public mlir::bmodelica::impl::ModelAlgorithmConversionPassBase<
          ModelAlgorithmConversionPass> {
public:
  using ModelAlgorithmConversionPassBase<
      ModelAlgorithmConversionPass>::ModelAlgorithmConversionPassBase;

  void runOnOperation() override;
};
} // namespace

void ModelAlgorithmConversionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<AlgorithmOp>([](AlgorithmOp op) {
    return !mlir::isa<DynamicOp, InitialOp>(op->getParentOp());
  });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::RewritePatternSet patterns(&getContext());

  // Counter for uniqueness of functions.
  size_t algorithmsCounter = 0;
  mlir::SymbolTableCollection symbolTable;

  patterns.insert<AlgorithmOpPattern>(&getContext(), symbolTable,
                                      algorithmsCounter);

  if (mlir::failed(
          applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createModelAlgorithmConversionPass() {
  return std::make_unique<ModelAlgorithmConversionPass>();
}
} // namespace mlir::bmodelica
