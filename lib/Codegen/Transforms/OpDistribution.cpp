#include "marco/Codegen/Transforms/OpDistribution.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  template<typename DistributableOp>
  class DistributionPass: public mlir::PassWrapper<DistributionPass<DistributableOp>, mlir::OperationPass<mlir::ModuleOp>>
  {
    public:
      void getDependentDialects(mlir::DialectRegistry& registry) const override
      {
        registry.insert<ModelicaDialect>();
      }

      void runOnOperation() override
      {
        auto module = this->getOperation();
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
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createNegateOpDistributionPass()
  {
    return std::make_unique<DistributionPass<NegateOp>>();
  }

  std::unique_ptr<mlir::Pass> createMulOpDistributionPass()
  {
    return std::make_unique<DistributionPass<MulOp>>();
  }

  std::unique_ptr<mlir::Pass> createDivOpDistributionPass()
  {
    return std::make_unique<DistributionPass<DivOp>>();
  }
}
