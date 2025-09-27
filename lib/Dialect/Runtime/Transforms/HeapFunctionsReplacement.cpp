#include "marco/Dialect/Runtime/Transforms/HeapFunctionsReplacement.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::runtime {
#define GEN_PASS_DEF_HEAPFUNCTIONSREPLACEMENTPASS
#include "marco/Dialect/Runtime/Transforms/Passes.h.inc"
} // namespace mlir::runtime

namespace {
class HeapFunctionsReplacementPass
    : public mlir::runtime::impl::HeapFunctionsReplacementPassBase<
          HeapFunctionsReplacementPass> {
public:
  using HeapFunctionsReplacementPassBase<
      HeapFunctionsReplacementPass>::HeapFunctionsReplacementPassBase;

  void runOnOperation() override;
};

std::string getReplacementFunctionName(llvm::StringRef functionName) {
  return "marco_" + functionName.str();
}

bool shouldBeReplaced(llvm::StringRef functionName) {
  return functionName == "malloc" || functionName == "realloc" ||
         functionName == "free";
}
} // namespace

void HeapFunctionsReplacementPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<mlir::Operation *> funcAndCallOps;

  moduleOp.walk([&](mlir::Operation *op) {
    if (mlir::isa<mlir::LLVM::LLVMFuncOp, mlir::LLVM::CallOp>(op)) {
      funcAndCallOps.push_back(op);
    }
  });

  mlir::parallelForEach(&getContext(), funcAndCallOps, [](mlir::Operation *op) {
    if (auto funcOp = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (auto symName = funcOp.getSymName(); shouldBeReplaced(symName)) {
        funcOp.setSymName(getReplacementFunctionName(symName));
      }

      return;
    }

    if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
      if (auto callee = callOp.getCallee();
          callee && shouldBeReplaced(*callee)) {
        callOp.setCallee(getReplacementFunctionName(*callee));
      }
    }
  });
}

namespace mlir::runtime {
std::unique_ptr<mlir::Pass> createHeapFunctionsReplacementPass() {
  return std::make_unique<HeapFunctionsReplacementPass>();
}
} // namespace mlir::runtime
