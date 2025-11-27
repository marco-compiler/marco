#include "marco/Frontend/Passes/AggressiveLICM.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"

namespace marco::frontend {
#define GEN_PASS_DEF_AGGRESSIVELICMPASS
#include "marco/Frontend/Passes.h.inc"
} // namespace marco::frontend

using namespace ::marco::frontend;

namespace {
class AggressiveLICMPass
    : public marco::frontend::impl::AggressiveLICMPassBase<AggressiveLICMPass> {
public:
  using AggressiveLICMPassBase::AggressiveLICMPassBase;

  void runOnOperation() override;
};
} // namespace

void AggressiveLICMPass::runOnOperation() {
  getOperation()->walk([](mlir::LoopLikeOpInterface loopLike) {
    mlir::moveLoopInvariantCode(
        loopLike.getLoopRegions(),
        [&](mlir::Value value, mlir::Region *) {
          return loopLike.isDefinedOutsideOfLoop(value);
        },
        [&](mlir::Operation *op, mlir::Region *) {
          auto memoryEffectInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);

          if (!memoryEffectInterface) {
            return false;
          }

          // Check if the operation has generic state-changing effects.
          llvm::SmallVector<
              mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
              effects;

          memoryEffectInterface.getEffectsOnResource(
              mlir::SideEffects::DefaultResource::get(), effects);

          return llvm::none_of(effects,
                               [](const mlir::SideEffects::EffectInstance<
                                   mlir::MemoryEffects::Effect> &effect) {
                                 return llvm::isa<mlir::MemoryEffects::Write>(
                                     effect.getEffect());
                               });
        },
        [&](mlir::Operation *op, mlir::Region *) {
          loopLike.moveOutOfLoop(op);
        });
  });
}

namespace marco::frontend {
std::unique_ptr<mlir::Pass> createAggressiveLICMPass() {
  return std::make_unique<AggressiveLICMPass>();
}
} // namespace marco::frontend
