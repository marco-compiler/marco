#include "marco/Dialect/Runtime/Transforms/HeapFunctionsReplacement.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
} // namespace

static std::string getReplacementFunctionName(llvm::StringRef functionName) {
  return "marco_" + functionName.str();
}

namespace {
class FunctionPattern : public mlir::OpRewritePattern<mlir::LLVM::LLVMFuncOp> {
public:
  FunctionPattern(llvm::StringRef functionName, mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::LLVM::LLVMFuncOp>(context),
        functionName(functionName.str()) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::LLVMFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getSymName() != functionName) {
      return rewriter.notifyMatchFailure(op, "Incompatible function");
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op.setSymName(getReplacementFunctionName(functionName)); });

    return mlir::success();
  }

private:
  std::string functionName;
};

struct MallocFunctionPattern : public FunctionPattern {
  explicit MallocFunctionPattern(mlir::MLIRContext *context)
      : FunctionPattern("malloc", context) {}
};

struct ReallocFunctionPattern : public FunctionPattern {
  explicit ReallocFunctionPattern(mlir::MLIRContext *context)
      : FunctionPattern("realloc", context) {}
};

struct FreeFunctionPattern : public FunctionPattern {
  explicit FreeFunctionPattern(mlir::MLIRContext *context)
      : FunctionPattern("free", context) {}
};

class CallPattern : public mlir::OpRewritePattern<mlir::LLVM::CallOp> {
public:
  CallPattern(llvm::StringRef functionName, mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::LLVM::CallOp>(context),
        functionName(functionName.str()) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto callee = op.getCallee();

    if (!callee) {
      return rewriter.notifyMatchFailure(op, "No callee specified");
    }

    if (callee != functionName) {
      return rewriter.notifyMatchFailure(op, "Incompatible function");
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op.setCallee(getReplacementFunctionName(functionName)); });

    return mlir::success();
  }

private:
  std::string functionName;
};

struct MallocCallPattern : public CallPattern {
  explicit MallocCallPattern(mlir::MLIRContext *context)
      : CallPattern("malloc", context) {}
};

struct ReallocCallPattern : public CallPattern {
  explicit ReallocCallPattern(mlir::MLIRContext *context)
      : CallPattern("realloc", context) {}
};

struct FreeCallPattern : public CallPattern {
  explicit FreeCallPattern(mlir::MLIRContext *context)
      : CallPattern("free", context) {}
};
} // namespace

void HeapFunctionsReplacementPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>(
      [](mlir::LLVM::LLVMFuncOp op) {
        auto functionName = op.getSymName();

        return functionName != "malloc" && functionName != "free" &&
               functionName != "realloc";
      });

  target.addDynamicallyLegalOp<mlir::LLVM::CallOp>([](mlir::LLVM::CallOp op) {
    auto callee = op.getCallee();

    if (!callee) {
      return true;
    }

    return callee != "malloc" && callee != "free" && callee != "realloc";
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<MallocFunctionPattern, ReallocFunctionPattern,
                  FreeFunctionPattern, MallocCallPattern, ReallocCallPattern,
                  FreeCallPattern>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
                                                std::move(patterns)))) {
    return signalPassFailure();
  }
}

namespace mlir::runtime {
std::unique_ptr<mlir::Pass> createHeapFunctionsReplacementPass() {
  return std::make_unique<HeapFunctionsReplacementPass>();
}
} // namespace mlir::runtime
