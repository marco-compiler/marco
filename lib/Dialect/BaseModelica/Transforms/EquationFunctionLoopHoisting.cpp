#include "marco/Dialect/BaseModelica/Transforms/EquationFunctionLoopHoisting.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_EQUATIONFUNCTIONLOOPHOISTINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class EquationFunctionLoopHoistingPass
      : public impl::EquationFunctionLoopHoistingPassBase<
            EquationFunctionLoopHoistingPass>
  {
    public:
      using EquationFunctionLoopHoistingPassBase
          ::EquationFunctionLoopHoistingPassBase;

      void runOnOperation() override;
  };
}

void EquationFunctionLoopHoistingPass::runOnOperation()
{
  size_t licmOps = 0;

  do {
    //mlir::bufferization::hoistBuffersFromLoops(getOperation());

    getOperation()->walk([&](mlir::LoopLikeOpInterface loopLike) {
      licmOps = moveLoopInvariantCode(
          loopLike.getLoopRegions(),
          [&](mlir::Value value, mlir::Region*) {
            return loopLike.isDefinedOutsideOfLoop(value);
          },
          [&](mlir::Operation* op, mlir::Region*) {
            return true;
          },
          [&](mlir::Operation* op, mlir::Region*) {
            loopLike.moveOutOfLoop(op);
          });
    });
  } while (licmOps != 0);
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createEquationFunctionLoopHoistingPass()
  {
    return std::make_unique<EquationFunctionLoopHoistingPass>();
  }
}
