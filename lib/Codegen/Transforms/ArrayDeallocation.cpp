#include "marco/Codegen/Transforms/ArrayDeallocation.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

namespace mlir::bmodelica
{
#define GEN_PASS_DEF_ARRAYDEALLOCATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class ArrayDeallocationPass
      : public mlir::bmodelica::impl::ArrayDeallocationPassBase<
          ArrayDeallocationPass>
  {
    public:
      using ArrayDeallocationPassBase<ArrayDeallocationPass>
          ::ArrayDeallocationPassBase;

      void runOnOperation() override;
  };
}

static void filterAndCollectOp(
    llvm::SmallVectorImpl<mlir::Operation*>& ops,
    mlir::Operation* op)
{
  if (mlir::isa<FunctionOp>(op)) {
    ops.push_back(op);
    return;
  }

  if (mlir::isa<EquationFunctionOp>(op)) {
    ops.push_back(op);
    return;
  }
}

void ArrayDeallocationPass::runOnOperation()
{
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<mlir::Operation*> ops;

  moduleOp.walk([&](mlir::Operation* op) {
    filterAndCollectOp(ops, op);
  });

  for (mlir::Operation* op : ops) {
    if (mlir::failed(mlir::bufferization::deallocateBuffers(op))) {
      return signalPassFailure();
    }
  }
}

namespace mlir::bmodelica
{
  std::unique_ptr<mlir::Pass> createArrayDeallocationPass()
  {
    return std::make_unique<ArrayDeallocationPass>();
  }
}
