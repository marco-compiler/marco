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
      void run(mlir::ModuleOp module)
      {
        llvm::SmallVector<DistributableOp> distributableOps;

        module.walk([&](DistributableOp op) {
          distributableOps.push_back(op);
        });

        for (auto op : distributableOps) {
          mlir::OpBuilder builder(op);

          auto distributableOp = mlir::cast<DistributableOpInterface>(op.getOperation());
          mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

          if (result != op) {
            op->replaceAllUsesWith(result);
          }
        }
      }
  };

  class NegateOpDistributionPass : public mlir::modelica::impl::NegateOpDistributionPassBase<NegateOpDistributionPass>, DistributionPass<NegateOp>
  {
    void runOnOperation() override
    {
      run(getOperation());
    }
  };

  class MulOpDistributionPass : public mlir::modelica::impl::MulOpDistributionPassBase<MulOpDistributionPass>, DistributionPass<MulOp>
  {
    void runOnOperation() override
    {
      run(getOperation());
    }
  };

  class DivOpDistributionPass : public mlir::modelica::impl::DivOpDistributionPassBase<DivOpDistributionPass>, DistributionPass<DivOp>
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
