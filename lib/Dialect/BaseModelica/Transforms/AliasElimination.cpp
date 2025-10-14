#include "marco/Dialect/BaseModelica/Transforms/AliasElimination.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_ALIASELIMINATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class AliasEliminationPass
    : public mlir::bmodelica::impl::AliasEliminationPassBase<
          AliasEliminationPass> {
public:
  using AliasEliminationPassBase::AliasEliminationPassBase;

  void runOnOperation() override;
};
} // namespace

void AliasEliminationPass::runOnOperation() {}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createAliasEliminationPass() {
  return std::make_unique<AliasEliminationPass>();
}
} // namespace mlir::bmodelica
