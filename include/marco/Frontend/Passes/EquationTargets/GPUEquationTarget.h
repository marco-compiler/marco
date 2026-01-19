#ifndef MARCO_FRONTEND_PASSES_EQUATIONTARGETS_GPUEQUATIONTARGET_H
#define MARCO_FRONTEND_PASSES_EQUATIONTARGETS_GPUEQUATIONTARGET_H

#include "marco/Frontend/Passes/EquationTargets/EquationTarget.h"

namespace marco::frontend {
class GPUEquationTarget : public EquationTarget {
public:
  explicit GPUEquationTarget(llvm::StringRef name);

  std::unique_ptr<EquationTarget> clone() const override;

  bool isCompatible(mlir::func::FuncOp equationFunction) const override;

  uint64_t getCost(mlir::func::FuncOp equationFunction) const override;

  mlir::transform::SequenceOp
  createTransformSequence(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                          mlir::Value moduleTransformValue) const override;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_EQUATIONTARGETS_GPUEQUATIONTARGET_H
