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
  };
}

template<typename EquationType>
static void eraseOldEquations(const Model<MatchedEquation>& model)
{
  llvm::DenseSet<mlir::Operation*> toBeKept;

  for (const auto& equation : model.getEquations()) {
    toBeKept.insert(equation->getOperation().getOperation());
  }

  llvm::DenseSet<mlir::Operation*> toBeErased;

  model.getOperation().template walk([&](EquationType equationOp) {
    if (mlir::Operation* op = equationOp.getOperation(); !toBeKept.contains(op)) {
      toBeErased.insert(op);
    }
  });

  for (mlir::Operation* op : toBeErased) {
    auto equation = Equation::build(mlir::cast<EquationInterface>(op), model.getVariables());
    equation->eraseIR();
  }
}

mlir::LogicalResult CyclesSolvingPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
{
  // The options to be used when printing the IR.
  ModelSolvingIROptions irOptions;
  irOptions.mergeAndSortRanges = debugView;
  irOptions.singleMatchAttr = debugView;
  irOptions.singleScheduleAttr = debugView;

  modelOp.dump();

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
    if (auto res = solveCycles(builder, matchedModel); mlir::failed(res)) {
      return res;
    }

    eraseOldEquations<InitialEquationOp>(matchedModel);

    // Write the match information in form of attributes.
    writeMatchingAttributes(builder, matchedModel, irOptions);
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
    if (auto res = solveCycles(builder, matchedModel); mlir::failed(res)) {
      return res;
    }

    eraseOldEquations<EquationOp>(matchedModel);

    // Write the match information in form of attributes.
    writeMatchingAttributes(builder, matchedModel, irOptions);
  }

  return mlir::success();
}

mlir::LogicalResult CyclesSolvingPass::solveCycles(mlir::OpBuilder& builder, Model<MatchedEquation>& model)
{
  if (auto res = ::solveCycles(model, builder); mlir::failed(res)) {
    if (solver.getKind() != Solver::Kind::ida) {
      // Check if the selected solver can deal with cycles. If not, fail.
      return res;
    }
  }

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
