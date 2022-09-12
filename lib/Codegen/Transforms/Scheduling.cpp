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
    Model<Equation> initialConditionsModel(modelOp);
    initialConditionsModel.setDerivativesMap(derivativesMap);

    // Discover variables and equations belonging to the 'initial conditions' model.
    initialConditionsModel.setVariables(discoverVariables(initialConditionsModel.getOperation()));
    initialConditionsModel.setEquations(discoverInitialEquations(initialConditionsModel.getOperation(), initialConditionsModel.getVariables()));

    // Obtain the matched model.
    Model<MatchedEquation> matchedInitialConditionsModel(modelOp);

    if (auto res = readMatchingAttributes(initialConditionsModel, matchedInitialConditionsModel); mlir::failed(res)) {
      return res;
    }

    // Compute the scheduled 'initial conditions' model.
    Model<ScheduledEquationsBlock> scheduledInitialConditionsModel(initialConditionsModel.getOperation());

    if (auto res = schedule(scheduledInitialConditionsModel, matchedInitialConditionsModel); mlir::failed(res)) {
      initialConditionsModel.getOperation().emitError("Scheduling failed for the 'initial conditions' model");
      return res;
    }

    // Write the schedule information in form of attributes.
    writeSchedulingAttributes(builder, scheduledInitialConditionsModel, irOptions);
  }

  if (processMainModel) {
    Model<Equation> mainModel(modelOp);
    mainModel.setDerivativesMap(derivativesMap);

    // Discover variables and equations belonging to the 'main' model.
    mainModel.setVariables(discoverVariables(mainModel.getOperation()));
    mainModel.setEquations(discoverEquations(mainModel.getOperation(), mainModel.getVariables()));

    // Obtain the matched model.
    Model<MatchedEquation> matchedMainModel(modelOp);

    if (auto res = readMatchingAttributes(mainModel, matchedMainModel); mlir::failed(res)) {
      return res;
    }

    // Compute the scheduled 'main' model.
    Model<ScheduledEquationsBlock> scheduledMainModel(mainModel.getOperation());

    if (auto res = schedule(scheduledMainModel, matchedMainModel); mlir::failed(res)) {
      mainModel.getOperation().emitError("Scheduling failed for the 'main' model");
      return res;
    }

    // Write the schedule information in form of attributes.
    writeSchedulingAttributes(builder, scheduledMainModel, irOptions);
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
