#include "marco/Codegen/Transforms/Scheduling.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_SCHEDULINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  class SchedulingPass : public mlir::modelica::impl::SchedulingPassBase<SchedulingPass>
  {
    public:
      using SchedulingPassBase::SchedulingPassBase;

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

mlir::LogicalResult SchedulingPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
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

    // Obtain the matched model.
    Model<MatchedEquation> matchedModel(modelOp);
    matchedModel.setVariables(model.getVariables());
    matchedModel.setDerivativesMap(model.getDerivativesMap());

    auto equationsFilter = [](EquationInterface op) {
      return mlir::isa<InitialEquationOp>(op);
    };

    if (auto res = readMatchingAttributes(matchedModel, equationsFilter); mlir::failed(res)) {
      return res;
    }

    // Compute the scheduled 'initial conditions' model.
    Model<ScheduledEquationsBlock> scheduledModel(matchedModel.getOperation());
    scheduledModel.setVariables(matchedModel.getVariables());
    scheduledModel.setDerivativesMap(matchedModel.getDerivativesMap());

    if (auto res = schedule(scheduledModel, matchedModel); mlir::failed(res)) {
      matchedModel.getOperation().emitError("Scheduling failed for the 'initial conditions' model");
      return res;
    }

    // Write the schedule information in form of attributes.
    writeSchedulingAttributes(builder, scheduledModel, irOptions);
  }

  if (processMainModel) {
    Model<Equation> model(modelOp);
    model.setDerivativesMap(derivativesMap);

    // Discover variables and equations belonging to the 'main' model.
    model.setVariables(discoverVariables(model.getOperation()));
    model.setEquations(discoverEquations(model.getOperation(), model.getVariables()));

    // Obtain the matched model.
    Model<MatchedEquation> matchedModel(modelOp);
    matchedModel.setVariables(model.getVariables());
    matchedModel.setDerivativesMap(model.getDerivativesMap());

    auto equationsFilter = [](EquationInterface op) {
      return mlir::isa<EquationOp>(op);
    };

    if (auto res = readMatchingAttributes(matchedModel, equationsFilter); mlir::failed(res)) {
      return res;
    }

    // Compute the scheduled 'main' model.
    Model<ScheduledEquationsBlock> scheduledModel(matchedModel.getOperation());
    scheduledModel.setVariables(matchedModel.getVariables());
    scheduledModel.setDerivativesMap(matchedModel.getDerivativesMap());

    if (auto res = schedule(scheduledModel, matchedModel); mlir::failed(res)) {
      matchedModel.getOperation().emitError("Scheduling failed for the 'main' model");
      return res;
    }

    // Write the schedule information in form of attributes.
    writeSchedulingAttributes(builder, scheduledModel, irOptions);
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createSchedulingPass()
  {
    return std::make_unique<SchedulingPass>();
  }

  std::unique_ptr<mlir::Pass> createSchedulingPass(const SchedulingPassOptions& options)
  {
    return std::make_unique<SchedulingPass>(options);
  }
}
