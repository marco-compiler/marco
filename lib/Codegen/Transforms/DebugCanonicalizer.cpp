#include "marco/Codegen/Transforms/DebugCanonicalizer.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Codegen/Transforms/ModelSolving/FoldingUtils.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_DEBUGCANONICALIZERPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  class DebugCanonicalizerPass : public mlir::modelica::impl::DebugCanonicalizerPassBase<DebugCanonicalizerPass>
  {
    public:
    using DebugCanonicalizerPassBase::DebugCanonicalizerPassBase;

    void runOnOperation() override
    {
      mlir::OpBuilder builder(getOperation());
      llvm::SmallVector<ModelOp, 1> modelOps;

      getOperation()->walk([&](ModelOp modelOp) {
        if (modelOp.getSymName() == modelName) {
          modelOps.push_back(modelOp);
        }
      });

      for (const auto& modelOp : modelOps) {
        if (mlir::failed(processModelOp(builder, modelOp))) {
          return signalPassFailure();
        }
      }
    }

    private:
    mlir::LogicalResult processModelOp(mlir::OpBuilder& builder, ModelOp modelOp);
  };
}

mlir::LogicalResult DebugCanonicalizerPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
{
  // The options to be used when printing the IR.
  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = debugView;
  irOptions.singleMatchAttr = debugView;
  irOptions.singleScheduleAttr = debugView;

  if (processICModel) {
    // Obtain the matched model.
    Model<MatchedEquation> matchedModel(modelOp);
    matchedModel.setVariables(discoverVariables(modelOp));

    auto equationsFilter = [](EquationInterface op) {
      return mlir::isa<InitialEquationOp>(op);
    };

    if (auto res = readMatchingAttributes(matchedModel, equationsFilter); mlir::failed(res)) {
      return res;
    }

    // Solve the cycles in the 'initial conditions' model.
    for (const auto& equation : matchedModel.getEquations()) {
      EquationSidesOp terminator = mlir::cast<EquationSidesOp>(equation->getOperation().bodyBlock()->getTerminator());
      llvm::dbgs() << "INDEX " << equation->getFlatAccessIndex(Point(0)) << ": " << recursiveFoldValue(terminator.getRhsValues()[0].getDefiningOp()) << "\n";
    }
  }

  if (processMainModel) {
    // Obtain the matched model.
    Model<MatchedEquation> matchedModel(modelOp);
    matchedModel.setVariables(discoverVariables(modelOp));

    auto equationsFilter = [](EquationInterface op) {
      return mlir::isa<EquationOp>(op);
    };

    if (auto res = readMatchingAttributes(matchedModel, equationsFilter); mlir::failed(res)) {
      return res;
    }

    // Solve the cycles in the 'main' model
    for (const auto& equation : matchedModel.getEquations()) {
      EquationSidesOp terminator = mlir::cast<EquationSidesOp>(equation->getOperation().bodyBlock()->getTerminator());
      llvm::dbgs() << "INDEX " << equation->getFlatAccessIndex(Point(0)) << ": " << recursiveFoldValue(terminator.getRhsValues()[0].getDefiningOp()) << "\n";
    }
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createDebugCanonicalizerPass()
  {
    return std::make_unique<DebugCanonicalizerPass>();
  }

  std::unique_ptr<mlir::Pass> createDebugCanonicalizerPass(const DebugCanonicalizerPassOptions& options)
  {
    return std::make_unique<DebugCanonicalizerPass>(options);
  }
}

