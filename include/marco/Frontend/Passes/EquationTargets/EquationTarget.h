#ifndef MARCO_FRONTEND_PASSES_EQUATIONTARGETS_EQUATIONTARGET_H
#define MARCO_FRONTEND_PASSES_EQUATIONTARGETS_EQUATIONTARGET_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

namespace marco::frontend {
class EquationTarget {
  std::string name;

protected:
  explicit EquationTarget(llvm::StringRef name);

public:
  virtual ~EquationTarget();

  virtual std::unique_ptr<EquationTarget> clone() const = 0;

  llvm::StringRef getName() const;

  virtual bool isCompatible(mlir::func::FuncOp equationFunction) const = 0;

  virtual uint64_t getCost(mlir::func::FuncOp equationFunction) const = 0;

  virtual mlir::transform::SequenceOp
  createTransformSequence(mlir::OpBuilder &builder,
                          mlir::ModuleOp moduleOp,
                          mlir::Value moduleTransformValue) const = 0;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_PASSES_EQUATIONTARGETS_EQUATIONTARGET_H
