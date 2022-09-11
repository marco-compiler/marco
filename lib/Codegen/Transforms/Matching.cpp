#include "marco/Codegen/Transforms/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_MATCHINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

static mlir::LogicalResult matchInitialConditionsModel(
    mlir::OpBuilder& builder,
    Model<MatchedEquation>& matchedModel,
    const Model<Equation>& model)
{
  auto matchableIndicesFn = [&](const Variable& variable) -> IndexSet {
    IndexSet matchableIndices(variable.getIndices());

    if (variable.isConstant()) {
      matchableIndices.clear();
      return matchableIndices;
    }

    return matchableIndices;
  };

  return match(matchedModel, model, matchableIndicesFn);
}

static mlir::LogicalResult matchMainModel(
    mlir::OpBuilder& builder,
    Model<MatchedEquation>& matchedModel,
    const Model<Equation>& model)
{
  auto matchableIndicesFn = [&](const Variable& variable) -> IndexSet {
    IndexSet matchableIndices(variable.getIndices());

    if (variable.isConstant()) {
      matchableIndices.clear();
      return matchableIndices;
    }

    auto argNumber = variable.getValue().cast<mlir::BlockArgument>().getArgNumber();

    // Remove the derived indices
    if (auto derivativesMap = matchedModel.getDerivativesMap(); derivativesMap.hasDerivative(argNumber)) {
      matchableIndices -= matchedModel.getDerivativesMap().getDerivedIndices(argNumber);
    }

    return matchableIndices;
  };

  return match(matchedModel, model, matchableIndicesFn);
}

namespace
{
  class MatchingPass : public mlir::modelica::impl::MatchingPassBase<MatchingPass>
  {
    public:
      using MatchingPassBase::MatchingPassBase;

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

mlir::LogicalResult MatchingPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
{
  Model<Equation> initialConditionsModel(modelOp);
  Model<Equation> mainModel(modelOp);

  // Retrieve the derivatives map computed by the legalization pass
  DerivativesMap derivativesMap;

  if (auto res = readDerivativesMap(modelOp, derivativesMap); mlir::failed(res)) {
    return res;
  }

  initialConditionsModel.setDerivativesMap(derivativesMap);
  mainModel.setDerivativesMap(derivativesMap);

  // Discover variables and equations belonging to the 'initial' model
  initialConditionsModel.setVariables(discoverVariables(initialConditionsModel.getOperation()));
  initialConditionsModel.setEquations(discoverInitialEquations(initialConditionsModel.getOperation(), initialConditionsModel.getVariables()));

  // Discover variables and equations belonging to the 'main' model
  mainModel.setVariables(discoverVariables(mainModel.getOperation()));
  mainModel.setEquations(discoverEquations(mainModel.getOperation(), mainModel.getVariables()));

  // Compute the matched 'initial' model
  Model<MatchedEquation> matchedInitialConditionsModel(initialConditionsModel.getOperation());

  if (auto res = matchInitialConditionsModel(builder, matchedInitialConditionsModel, initialConditionsModel); mlir::failed(res)) {
    initialConditionsModel.getOperation().emitError("Matching failed for the 'initial conditions' model");
    return res;
  }

  // Compute the matched 'main' model
  Model<MatchedEquation> matchedMainModel(mainModel.getOperation());

  if (auto res = matchMainModel(builder, matchedMainModel, mainModel); mlir::failed(res)) {
    mainModel.getOperation().emitError("Matching failed for the 'main' model");
    return res;
  }

  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = mergeAndSortRanges;

  writeMatchingAttributes(builder, matchedInitialConditionsModel, irOptions);
  writeMatchingAttributes(builder, matchedMainModel, irOptions);

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createMatchingPass()
  {
    return std::make_unique<MatchingPass>();
  }

  std::unique_ptr<mlir::Pass> createMatchingPass(const MatchingPassOptions& options)
  {
    return std::make_unique<MatchingPass>(options);
  }
}
