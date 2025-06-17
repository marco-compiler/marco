#ifndef MARCO_JIT_ENGINEWRAPPER_H
#define MARCO_JIT_ENGINEWRAPPER_H

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"

namespace marco::jit {
class EngineBuilder {
  mlir::ModuleOp moduleOp;
  mlir::ExecutionEngineOptions options;

public:
  EngineBuilder(mlir::ModuleOp moduleOp,
                const mlir::ExecutionEngineOptions &options = {});

  std::unique_ptr<mlir::ExecutionEngine> getEngine() const;

private:
  mlir::ModuleOp lowerToLLVM() const;
};
} // namespace marco::jit

#endif // MARCO_JIT_ENGINEWRAPPER_H
