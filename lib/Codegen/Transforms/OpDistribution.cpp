#include "marco/Codegen/Transforms/OpDistribution.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include <stack>

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
        std::stack<mlir::Operation*> touched;

        module.walk([&](DistributableOp op) {
          mlir::OpBuilder builder(op);

          auto distributableOp = mlir::cast<DistributableOpInterface>(op.getOperation());
          mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

          if (result != op) {
            op->replaceAllUsesWith(result);
          }

          touched.push(op.getOperation());
        });

        // Erase the operation whose results are unused, for easier testing
        while (!touched.empty()) {
          auto op = touched.top();
          touched.pop();

          if (op->getUsers().empty()) {
            for (const auto& operand : op->getOperands()) {
              if (auto definingOp = operand.getDefiningOp()) {
                touched.push(definingOp);
              }
            }

            op->erase();
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
