#include "marco/Frontend/Passes/EquationTargets/GPUEquationTarget.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

using namespace ::marco::frontend;

namespace {
bool isParallelizable(mlir::func::FuncOp equationFunction) {
  auto hasParallelizableAttr =
      equationFunction->getAttrOfType<mlir::BoolAttr>("parallelizable");

  return hasParallelizableAttr && hasParallelizableAttr.getValue();
}

size_t getNumOfPerfectlyNestedLoops(mlir::affine::AffineForOp forOp) {
  llvm::SmallVector<mlir::affine::AffineForOp> nestedLoops;
  mlir::affine::getPerfectlyNestedLoops(nestedLoops, forOp);
  return nestedLoops.size();
}

size_t getNumOfPerfectlyNestedLoops(mlir::func::FuncOp funcOp) {
  size_t maxCount = 0;

  for (auto &region : funcOp.getBody()) {
    for (mlir::affine::AffineForOp forOp :
         region.getOps<mlir::affine::AffineForOp>()) {
      size_t count = getNumOfPerfectlyNestedLoops(forOp);
      maxCount = std::max(maxCount, count);
    }
  }

  return maxCount;
}
} // namespace

namespace marco::frontend {
GPUEquationTarget::GPUEquationTarget(llvm::StringRef name)
    : EquationTarget(name) {}

std::unique_ptr<EquationTarget> GPUEquationTarget::clone() const {
  return std::make_unique<GPUEquationTarget>(*this);
}

bool GPUEquationTarget::isCompatible(
    mlir::func::FuncOp equationFunction) const {
  if (!isParallelizable(equationFunction)) {
    return false;
  }

  int numNestedLoops = getNumOfPerfectlyNestedLoops(equationFunction);
  return numNestedLoops >= 1;
}

uint64_t GPUEquationTarget::getCost(mlir::func::FuncOp equationFunction) const {
  // Until a proper cost model is implemented, prefer GPU target by returning a
  // low cost.
  return 1;
}

mlir::transform::SequenceOp GPUEquationTarget::createTransformSequence(
    mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
    mlir::Value moduleTransformValue) const {
  return builder.create<mlir::transform::SequenceOp>(
      moduleOp.getLoc(), mlir::TypeRange{},
      mlir::transform::FailurePropagationMode::Propagate, moduleTransformValue,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::BlockArgument arg) {
        mlir::Value funcOp = arg;

        // Raise memref accesses to affine accesses.
        funcOp = nestedBuilder.create<mlir::transform::ApplyRegisteredPassOp>(
            loc, funcTransformValue.getType(), funcOp, "access-affine-raise",
            nullptr, mlir::ValueRange());

        // Execute the body of the equation only if the indices are within
        // bounds.
        funcOp = nestedBuilder.create<mlir::transform::ApplyRegisteredPassOp>(
            loc, funcTransformValue.getType(), funcOp,
            "equation-index-check-insertion", nullptr, mlir::ValueRange());

        // Convert affine to gpu.
        funcOp = nestedBuilder.create<mlir::transform::ApplyRegisteredPassOp>(
            loc, funcTransformValue.getType(), funcOp,
            "convert-affine-for-to-gpu", nullptr, mlir::ValueRange());

        // Expand strided metadata.
        nestedBuilder.create<mlir::transform::ApplyRegisteredPassOp>(
            loc, funcTransformValue.getType(), funcOp,
            "expand-strided-metadata", nullptr, mlir::ValueRange());

        nestedBuilder.create<mlir::transform::YieldOp>(loc);
      });
}
} // namespace marco::frontend
