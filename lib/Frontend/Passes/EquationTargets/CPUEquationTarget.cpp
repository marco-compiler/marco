#include "marco/Frontend/Passes/EquationTargets/CPUEquationTarget.h"

using namespace ::marco::frontend;

namespace marco::frontend {
CPUEquationTarget::CPUEquationTarget(llvm::StringRef name)
    : EquationTarget(name) {}

std::unique_ptr<EquationTarget> CPUEquationTarget::clone() const {
  return std::make_unique<CPUEquationTarget>(*this);
}

bool CPUEquationTarget::isCompatible(
    mlir::func::FuncOp equationFunction) const {
  // The CPU target is compatible with all functions.
  return true;
}

uint64_t CPUEquationTarget::getCost(mlir::func::FuncOp equationFunction) const {
  // Until a proper cost model is implemented, return the maximum cost.
  return std::numeric_limits<uint64_t>::max();
}

mlir::transform::SequenceOp CPUEquationTarget::createTransformSequence(
    mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
    mlir::Value moduleTransformValue) const {
  return builder.create<mlir::transform::SequenceOp>(
      moduleOp.getLoc(), mlir::TypeRange{},
      mlir::transform::FailurePropagationMode::Propagate, moduleTransformValue,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::BlockArgument) {
        nestedBuilder.create<mlir::transform::YieldOp>(loc);
      });
}
} // namespace marco::frontend
