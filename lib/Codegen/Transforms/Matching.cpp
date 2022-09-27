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

    // Remove the derived indices.
    const DerivativesMap& derivativesMap = matchedModel.getDerivativesMap();

    if (derivativesMap.hasDerivative(argNumber)) {
      matchableIndices -= derivativesMap.getDerivedIndices(argNumber);
    }

    return matchableIndices;
  };

  return match(matchedModel, model, matchableIndicesFn);
}

static void splitIndices(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
{
  Equations<MatchedEquation> equations;

  for (const auto& equation : model.getEquations()) {
    auto write = equation->getWrite();
    auto iterationRanges = equation->getIterationRanges();
    auto writtenIndices = write.getAccessFunction().map(iterationRanges);

    IndexSet result;

    for (const auto& access : equation->getAccesses()) {
      if (access.getPath() == write.getPath()) {
        continue;
      }

      if (access.getVariable() != write.getVariable()) {
        continue;
      }

      auto accessedIndices = access.getAccessFunction().map(iterationRanges);

      if (!accessedIndices.overlaps(writtenIndices)) {
        continue;
      }

      result += write.getAccessFunction().inverseMap(
          IndexSet(accessedIndices.intersect(writtenIndices)),
          IndexSet(iterationRanges));
    }

    for (const auto& range : llvm::make_range(result.rangesBegin(), result.rangesEnd())) {
      auto clone = Equation::build(equation->getOperation(), equation->getVariables());

      auto matchedClone = std::make_unique<MatchedEquation>(
          std::move(clone), IndexSet(range), write.getPath());

      equations.add(std::move(matchedClone));
    }

    auto values = (IndexSet(iterationRanges) - result);
    for (const auto& range : llvm::make_range(values.rangesBegin(), values.rangesEnd()) ) {
      auto clone = Equation::build(equation->getOperation(), equation->getVariables());

      auto matchedClone = std::make_unique<MatchedEquation>(
          std::move(clone), IndexSet(range), write.getPath());

      equations.add(std::move(matchedClone));
    }
  }

  model.setEquations(equations);
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
  // The options to be used when printing the IR.
  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = debugView;
  irOptions.singleMatchAttr = debugView;
  irOptions.singleScheduleAttr = debugView;

  // Retrieve the derivatives map computed by the legalization pass.
  DerivativesMap derivativesMap;

  if (auto res = readDerivativesMap(modelOp, derivativesMap); mlir::failed(res)) {
    return res;
  }

  if (processICModel) {
    Model<Equation> model(modelOp);
    model.setDerivativesMap(derivativesMap);

    // Discover variables and equations belonging to the 'initial conditions' model.
    model.setVariables(discoverVariables(model.getOperation()));
    model.setEquations(discoverInitialEquations(model.getOperation(), model.getVariables()));

    // Compute the matched 'initial conditions' model.
    Model<MatchedEquation> matchedModel(model.getOperation());
    matchedModel.setDerivativesMap(model.getDerivativesMap());

    if (auto res = matchInitialConditionsModel(builder, matchedModel, model); mlir::failed(res)) {
      model.getOperation().emitError("Matching failed for the 'initial conditions' model");
      return res;
    }

    splitIndices(builder, matchedModel);

    // Write the match information in form of attributes.
    writeMatchingAttributes(builder, matchedModel, irOptions);
  }

  if (processMainModel) {
    Model<Equation> model(modelOp);
    model.setDerivativesMap(derivativesMap);

    // Discover variables and equations belonging to the 'main' model.
    model.setVariables(discoverVariables(model.getOperation()));
    model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

    // Compute the matched 'main' model.
    Model<MatchedEquation> matchedModel(model.getOperation());
    matchedModel.setDerivativesMap(model.getDerivativesMap());

    if (auto res = matchMainModel(builder, matchedModel, model); mlir::failed(res)) {
      model.getOperation().emitError("Matching failed for the 'main' model");
      return res;
    }

    splitIndices(builder, matchedModel);

    // Write the match information in form of attributes.
    writeMatchingAttributes(builder, matchedModel, irOptions);
  }

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
