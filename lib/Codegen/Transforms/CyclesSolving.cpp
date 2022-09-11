#include "marco/Codegen/Transforms/CyclesSolving.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Cycles.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_CYCLESSOLVINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  class CyclesSolvingPass : public mlir::modelica::impl::CyclesSolvingPassBase<CyclesSolvingPass>
  {
    public:
      using CyclesSolvingPassBase::CyclesSolvingPassBase;

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

      mlir::LogicalResult solveCycles(mlir::OpBuilder& builder, Model<MatchedEquation>& model);

      mlir::LogicalResult splitEquations(mlir::OpBuilder& builder, Model<MatchedEquation>& model);
  };
}

mlir::LogicalResult CyclesSolvingPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
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

  // Obtain the matched models
  Model<MatchedEquation> matchedInitialConditionsModel(modelOp);
  Model<MatchedEquation> matchedMainModel(modelOp);

  if (auto res = readMatchingAttributes(initialConditionsModel, matchedInitialConditionsModel); mlir::failed(res)) {
    return res;
  }

  if (auto res = readMatchingAttributes(mainModel, matchedMainModel); mlir::failed(res)) {
    return res;
  }

  // Solve the cycles in the 'initial conditions' model
  if (auto res = solveCycles(builder, matchedInitialConditionsModel); mlir::failed(res)) {
    return res;
  }

  // Solve the cycles in the 'main' model
  if (auto res = solveCycles(builder, matchedMainModel); mlir::failed(res)) {
    return res;
  }

  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = mergeAndSortRanges;

  writeMatchingAttributes(builder, matchedInitialConditionsModel, irOptions);
  writeMatchingAttributes(builder, matchedMainModel, irOptions);

  return mlir::success();
}

mlir::LogicalResult CyclesSolvingPass::solveCycles(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
{
  if (auto res = splitEquations(builder, model); mlir::failed(res)) {
    return res;
  }

  if (auto res = ::solveCycles(model, builder); mlir::failed(res)) {
    if (solver.getKind() != Solver::Kind::ida) {
      // Check if the selected solver can deal with cycles. If not, fail.
      return res;
    }
  }

  return mlir::success();
}

mlir::LogicalResult CyclesSolvingPass::splitEquations(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
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
  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createCyclesSolvingPass()
  {
    return std::make_unique<CyclesSolvingPass>();
  }

  std::unique_ptr<mlir::Pass> createCyclesSolvingPass(const CyclesSolvingPassOptions& options)
  {
    return std::make_unique<CyclesSolvingPass>(options);
  }
}
