#include "marco/Codegen/Transforms/OpDistribution.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::modelica
{
#define GEN_PASS_DEF_NEGATEOPDISTRIBUTIONPASS
#define GEN_PASS_DEF_MULOPDISTRIBUTIONPASS
#define GEN_PASS_DEF_DIVOPDISTRIBUTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

namespace
{
  template<typename DistributableOp>
  class DistributionPass
  {
    public:
      void run(mlir::ModuleOp moduleOp)
      {
        llvm::SmallVector<DistributableOp> distributableOps;

        moduleOp.walk([&](DistributableOp op) {
          distributableOps.push_back(op);
        });

        for (auto op : distributableOps) {
          mlir::OpBuilder builder(op);

          auto distributableOp =
              mlir::cast<DistributableOpInterface>(op.getOperation());

          llvm::SmallVector<mlir::Value> replacements;

          if (mlir::succeeded(distributableOp.distribute(
                  replacements, builder))) {
            for (size_t i = 0, e = distributableOp->getNumResults();
                 i < e; ++i) {
              mlir::Value oldValue = distributableOp->getResult(i);
              mlir::Value newValue = replacements[i];

              if (oldValue != newValue) {
                oldValue.replaceAllUsesWith(newValue);
              }
            }
          }
        }
      }
  };

  class NegateOpDistributionPass
      : public mlir::modelica::impl::NegateOpDistributionPassBase<
          NegateOpDistributionPass>,
          DistributionPass<NegateOp>
  {
    void runOnOperation() override
    {
      run(getOperation());
    }
  };

  class MulOpDistributionPass
      : public mlir::modelica::impl::MulOpDistributionPassBase<
          MulOpDistributionPass>,
          DistributionPass<MulOp>
  {
    void runOnOperation() override
    {
      run(getOperation());
    }
  };

  class DivOpDistributionPass
      : public mlir::modelica::impl::DivOpDistributionPassBase<
          DivOpDistributionPass>,
        DistributionPass<DivOp>
  {
    void runOnOperation() override
    {
      run(getOperation());
    }
  };
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createNegateOpDistributionPass()
  {
    return std::make_unique<NegateOpDistributionPass>();
  }

  std::unique_ptr<mlir::Pass> createMulOpDistributionPass()
  {
    return std::make_unique<MulOpDistributionPass>();
  }

  std::unique_ptr<mlir::Pass> createDivOpDistributionPass()
  {
    return std::make_unique<DivOpDistributionPass>();
  }
}
