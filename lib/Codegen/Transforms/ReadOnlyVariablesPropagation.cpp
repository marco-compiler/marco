#include "marco/Codegen/Transforms/ReadOnlyVariablesPropagation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_READONLYVARIABLESPROPAGATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  class ReadOnlyVariablesPropagationPass
      : public mlir::modelica::impl::ReadOnlyVariablesPropagationPassBase<
            ReadOnlyVariablesPropagationPass>
  {
    public:
      using ReadOnlyVariablesPropagationPassBase
        ::ReadOnlyVariablesPropagationPassBase;

      void runOnOperation() override
      {
        llvm::SmallVector<ModelOp, 1> modelOps;

        getOperation()->walk([&](ModelOp modelOp) {
          modelOps.push_back(modelOp);
        });

        for (ModelOp modelOp : modelOps) {
          if (mlir::failed(processModelOp(modelOp))) {
            return signalPassFailure();
          }
        }
      }

    private:
      llvm::DenseSet<llvm::StringRef> getIgnoredVariableNames() const;

      mlir::LogicalResult processModelOp(ModelOp modelOp);
  };
}

llvm::DenseSet<llvm::StringRef>
ReadOnlyVariablesPropagationPass::getIgnoredVariableNames() const
{
  llvm::DenseSet<llvm::StringRef> result;
  llvm::SmallVector<llvm::StringRef> names;
  llvm::StringRef(ignoredVariables).split(names, ',');
  result.insert(names.begin(), names.end());
  return result;
}

mlir::LogicalResult
ReadOnlyVariablesPropagationPass::processModelOp(ModelOp modelOp)
{
  mlir::SymbolTable symbolTable(modelOp);

  llvm::DenseSet<llvm::StringRef> nonPropagatableVariables =
      getIgnoredVariableNames();

  // Collect the regions into which the replacement has to be performed.
  llvm::SmallVector<mlir::Region*> regions;

  for (auto& op : modelOp.getOps()) {
    for (mlir::Region& region : op.getRegions()) {
      regions.push_back(&region);
    }
  }

  // Collect the binding equations for faster lookup.
  llvm::StringMap<BindingEquationOp> bindingEquations;

  for (BindingEquationOp bindingEquationOp :
       modelOp.getOps<BindingEquationOp>()) {
    bindingEquations[bindingEquationOp.getVariable()] = bindingEquationOp;
  }

  return mlir::failableParallelForEach(
      &getContext(), regions,
      [&symbolTable, &nonPropagatableVariables, &bindingEquations]
      (mlir::Region* region) {
        mlir::OpBuilder builder(region->getContext());
        llvm::SmallVector<VariableGetOp> getOps;

        region->walk([&](VariableGetOp getOp) {
          getOps.push_back(getOp);
        });

        for (VariableGetOp getOp : getOps) {
          if (nonPropagatableVariables.contains(getOp.getMember())) {
            // The variable has been explicitly set as ignored.
            continue;
          }

          // Check if there is a binding equation for the variable.
          auto bindingEquationIt = bindingEquations.find(getOp.getMember());

          if (bindingEquationIt == bindingEquations.end()) {
            continue;
          }

          BindingEquationOp bindingEquationOp = bindingEquationIt->getValue();

          // Get the variable declaration.
          auto variable = symbolTable.lookup<VariableOp>(getOp.getMember());

          if (!variable.isReadOnly()) {
            // Writable variables can't be propagated.
            continue;
          }

          // Clone the operations used to obtain the replacement value.
          builder.setInsertionPoint(getOp);
          mlir::BlockAndValueMapping mapping;

          for (auto& op : bindingEquationOp.getBodyRegion().getOps()) {
            if (!mlir::isa<YieldOp>(op)) {
              builder.clone(op, mapping);
            }
          }

          // Get the value to be used as replacement.
          auto yieldOp = mlir::cast<YieldOp>(
              bindingEquationOp.getBodyRegion().back().getTerminator());

          mlir::Value replacement = mapping.lookup(yieldOp.getValues()[0]);

          // Perform the replacement.
          getOp.replaceAllUsesWith(replacement);
          getOp->erase();
        }

        return mlir::success();
      });
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createReadOnlyVariablesPropagationPass()
  {
    return std::make_unique<ReadOnlyVariablesPropagationPass>();
  }

  std::unique_ptr<mlir::Pass> createReadOnlyVariablesPropagationPass(
      const ReadOnlyVariablesPropagationPassOptions& options)
  {
    return std::make_unique<ReadOnlyVariablesPropagationPass>(options);
  }
}
