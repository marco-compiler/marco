#include "marco/Codegen/Transforms/ConstantFolding.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "mlir/Transforms/FoldUtils.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_CONSTANTFOLDINGPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  class ConstantFoldingPass : public mlir::modelica::impl::ConstantFoldingPassBase<ConstantFoldingPass>
  {
    public:
    using ConstantFoldingPassBase::ConstantFoldingPassBase;

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

static void foldValue(EquationInterface equationOp, mlir::Value value)
{
  mlir::OperationFolder helper(value.getContext());
  std::stack<mlir::Operation*> visitStack;
  llvm::SmallVector<mlir::Operation*, 3> ops;

  if (auto definingOp = value.getDefiningOp()) {
    visitStack.push(definingOp);
  }

  while (!visitStack.empty()) {
    auto op = visitStack.top();
    visitStack.pop();

    ops.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        visitStack.push(definingOp);
      }
    }
  }

  llvm::SmallVector<mlir::Operation*, 3> constants;

  for (auto *op : llvm::reverse(ops)) {
    helper.tryToFold(op, [&](mlir::Operation* constant) {
      constants.push_back(constant);
    });
  }

  for (auto* op : llvm::reverse(constants)) {
    op->moveBefore(equationOp.bodyBlock(), equationOp.bodyBlock()->begin());
  }
}

mlir::LogicalResult ConstantFoldingPass::processModelOp(mlir::OpBuilder& builder, ModelOp modelOp)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::OperationFolder helper(builder.getContext());
  std::vector<EquationOp> equationOps;

  modelOp.getBodyRegion().walk([&](EquationOp equationOp) {
    equationOps.push_back(equationOp);
  });

  for(auto& equationOp : equationOps)
  {
    auto terminator = mlir::cast<EquationSidesOp>(
        equationOp.bodyBlock()->getTerminator());

    auto lhsValues = terminator.getLhsValues();
    auto rhsValues = terminator.getRhsValues();

    for(auto value : lhsValues)
      foldValue(equationOp, value);

    for(auto value : rhsValues)
      foldValue(equationOp, value);
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createConstantFoldingPass()
  {
    return std::make_unique<ConstantFoldingPass>();
  }

  std::unique_ptr<mlir::Pass> createConstantFoldingPass(const ConstantFoldingPassOptions& options)
  {
    return std::make_unique<ConstantFoldingPass>(options);
  }
}
