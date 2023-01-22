#include "marco/Codegen/Transforms/ParametersPropagation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Model.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/Dependency.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_PARAMETERSPROPAGATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace
{
  class ParametersPropagationPass
      : public mlir::modelica::impl::ParametersPropagationPassBase<
            ParametersPropagationPass>
  {
    public:
      using ParametersPropagationPassBase::ParametersPropagationPassBase;

      void runOnOperation() override
      {
        mlir::OpBuilder builder(getOperation());
        llvm::SmallVector<ModelOp, 1> modelOps;

        getOperation()->walk([&](ModelOp modelOp) {
          modelOps.push_back(modelOp);
        });

        for (ModelOp modelOp : modelOps) {
          if (mlir::failed(processModelOp(builder, modelOp))) {
            return signalPassFailure();
          }
        }
      }

    private:
      llvm::DenseSet<llvm::StringRef> getIgnoredParameterNames() const;

      mlir::LogicalResult processModelOp(
          mlir::OpBuilder& builder, ModelOp modelOp);
  };
}

llvm::DenseSet<llvm::StringRef>
ParametersPropagationPass::getIgnoredParameterNames() const
{
  llvm::DenseSet<llvm::StringRef> result;
  llvm::SmallVector<llvm::StringRef> names;
  llvm::StringRef(ignoredParameters).split(names, ',');
  result.insert(names.begin(), names.end());
  return result;
}

mlir::LogicalResult ParametersPropagationPass::processModelOp(
    mlir::OpBuilder& builder, ModelOp modelOp)
{
  llvm::DenseSet<llvm::StringRef> nonPropagatableParameters =
      getIgnoredParameterNames();

  auto memberCreateOps = modelOp.getVariableDeclarationOps();

  for (BindingEquationOp bindingEquationOp :
       modelOp.getBodyRegion().getOps<BindingEquationOp>()) {
    mlir::Value variable = bindingEquationOp.getVariable();

    unsigned int variableArgNumber =
        variable.cast<mlir::BlockArgument>().getArgNumber();

    MemberCreateOp memberCreateOp = memberCreateOps[variableArgNumber];

    if (!memberCreateOp.isParameter()) {
      continue;
    }

    if (nonPropagatableParameters.contains(memberCreateOp.getSymName())) {
      continue;
    }

    for (auto& use : llvm::make_early_inc_range(variable.getUses())) {
      mlir::Operation* user = use.getOwner();

      if (!user->getParentOfType<EquationInterface>()) {
        continue;
      }

      builder.setInsertionPoint(user);
      mlir::BlockAndValueMapping mapping;

      for (auto& op : bindingEquationOp.getBodyRegion().getOps()) {
        if (!mlir::isa<YieldOp>(op)) {
          builder.clone(op, mapping);
        }
      }

      auto yieldOp = mlir::cast<YieldOp>(
          bindingEquationOp.getBodyRegion().back().getTerminator());

      mlir::Value replacement = mapping.lookup(yieldOp.getValues()[0]);

      if (auto loadOp = mlir::dyn_cast<LoadOp>(user);
          loadOp && loadOp.getIndices().empty()) {
        loadOp.getResult().replaceAllUsesWith(replacement);
        user->erase();
      } else {
        use.set(replacement);
      }
    }
  }

  return mlir::success();
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createParametersPropagationPass()
  {
    return std::make_unique<ParametersPropagationPass>();
  }

  std::unique_ptr<mlir::Pass> createParametersPropagationPass(
      const ParametersPropagationPassOptions& options)
  {
    return std::make_unique<ParametersPropagationPass>(options);
  }
}
