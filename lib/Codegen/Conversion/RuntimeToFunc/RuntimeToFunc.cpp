#include "marco/Codegen/Conversion/RuntimeToFunc/RuntimeToFunc.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_RUNTIMETOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::runtime;

namespace {
/// Generic rewrite pattern that provides some utility functions.
template <typename Op>
class RuntimeOpRewritePattern : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

protected:
  mlir::Value createGlobalString(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::ModuleOp moduleOp, mlir::StringRef name,
                                 mlir::StringRef value) const {
    mlir::LLVM::GlobalOp global;

    {
      // Create the global at the entry of the module.
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(moduleOp.getBody());

      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(
              llvm::StringRef(value.data(), value.size() + 1)));
    }

    // Get the pointer to the first character of the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    mlir::Type type = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);

    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), type,
        globalPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
  }
};
} // namespace

namespace {
class VariableGetterOpLowering
    : public RuntimeOpRewritePattern<VariableGetterOp> {
public:
  using RuntimeOpRewritePattern<VariableGetterOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(VariableGetterOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Type ptrType =
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        loc, op.getSymName(),
        rewriter.getFunctionType(ptrType, op.getResultTypes()));

    mlir::IRMapping mapping;

    // Create the entry block.
    mlir::Block *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // Move the function block.
    size_t numOfIndices = op.getIndices().size();
    mlir::Block *firstSourceBlock = &op.getFunctionBody().front();

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    // Extract and map the indices.
    rewriter.setInsertionPointToStart(entryBlock);
    llvm::SmallVector<mlir::Value, 3> mappedIndices;

    for (size_t i = 0; i < numOfIndices; ++i) {
      mlir::Value address = rewriter.create<mlir::LLVM::GEPOp>(
          loc, ptrType, rewriter.getI64Type(), funcOp.getArgument(0),
          llvm::ArrayRef<mlir::LLVM::GEPArg>(static_cast<int32_t>(i)));

      mlir::Value index = rewriter.create<mlir::LLVM::LoadOp>(
          loc, rewriter.getI64Type(), address);

      index = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getIndexType(), index);

      mappedIndices.push_back(index);
    }

    // Branch to the moved body region.
    rewriter.create<mlir::cf::BranchOp>(op.getLoc(), firstSourceBlock,
                                        mappedIndices);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class InitFunctionOpLowering : public RuntimeOpRewritePattern<InitFunctionOp> {
public:
  using RuntimeOpRewritePattern<InitFunctionOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InitFunctionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "init", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    auto terminator =
        mlir::cast<YieldOp>(funcOp.getFunctionBody().back().getTerminator());

    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class DeinitFunctionOpLowering
    : public RuntimeOpRewritePattern<DeinitFunctionOp> {
public:
  using RuntimeOpRewritePattern<DeinitFunctionOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DeinitFunctionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "deinit", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    auto terminator =
        mlir::cast<YieldOp>(funcOp.getFunctionBody().back().getTerminator());

    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ICModelBeginOpLowering : public RuntimeOpRewritePattern<ICModelBeginOp> {
public:
  using RuntimeOpRewritePattern<ICModelBeginOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ICModelBeginOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "icModelBegin", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ICModelEndOpLowering : public RuntimeOpRewritePattern<ICModelEndOp> {
public:
  using RuntimeOpRewritePattern<ICModelEndOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ICModelEndOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "icModelEnd", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class DynamicModelBeginOpLowering
    : public RuntimeOpRewritePattern<DynamicModelBeginOp> {
public:
  using RuntimeOpRewritePattern<DynamicModelBeginOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DynamicModelBeginOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "dynamicModelBegin", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class DynamicModelEndOpLowering
    : public RuntimeOpRewritePattern<DynamicModelEndOp> {
public:
  using RuntimeOpRewritePattern<DynamicModelEndOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DynamicModelEndOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), "dynamicModelEnd", rewriter.getFunctionType({}, {}));

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
    rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class EquationFunctionOpLowering
    : public RuntimeOpRewritePattern<EquationFunctionOp> {
public:
  using RuntimeOpRewritePattern<EquationFunctionOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EquationFunctionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    llvm::SmallVector<mlir::Type, 1> argsTypes;

    argsTypes.push_back(
        mlir::LLVM::LLVMPointerType::get(rewriter.getContext()));

    auto functionType = rewriter.getFunctionType(argsTypes, {});

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getSymName(), functionType);

    mlir::Block *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    llvm::SmallVector<mlir::Value> mappedBoundaries;
    mlir::Value equationBoundariesPtr = funcOp.getArgument(0);

    for (auto arg : llvm::enumerate(op.getArguments())) {
      mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
          arg.value().getLoc(), rewriter.getI64IntegerAttr(arg.index()));

      mlir::Value boundaryPtr = rewriter.create<mlir::LLVM::GEPOp>(
          arg.value().getLoc(), equationBoundariesPtr.getType(),
          rewriter.getI64Type(), equationBoundariesPtr, index);

      mlir::Value mappedBoundary = rewriter.create<mlir::LLVM::LoadOp>(
          boundaryPtr.getLoc(), rewriter.getI64Type(), boundaryPtr);

      mappedBoundary = rewriter.create<mlir::arith::IndexCastOp>(
          mappedBoundary.getLoc(), rewriter.getIndexType(), mappedBoundary);

      mappedBoundaries.push_back(mappedBoundary);
    }

    rewriter.create<mlir::cf::BranchOp>(funcOp.getLoc(), &op.getBody().front(),
                                        mappedBoundaries);

    rewriter.inlineRegionBefore(op.getBody(), funcOp.getFunctionBody(),
                                funcOp.end());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class FunctionOpLowering : public RuntimeOpRewritePattern<FunctionOp> {
public:
  using RuntimeOpRewritePattern<FunctionOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(FunctionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.isDeclaration()) {
      return rewriter.notifyMatchFailure(op, "Declaration");
    }

    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto funcOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getSymName(), op.getFunctionType());

    rewriter.inlineRegionBefore(op.getBodyRegion(), funcOp.getFunctionBody(),
                                funcOp.getFunctionBody().end());

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReturnOpLowering : public RuntimeOpRewritePattern<ReturnOp> {
  using RuntimeOpRewritePattern<ReturnOp>::RuntimeOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReturnOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op.getOperands());
    return mlir::success();
  }
};
} // namespace

namespace {
class RuntimeToFuncConversionPass
    : public mlir::impl::RuntimeToFuncConversionPassBase<
          RuntimeToFuncConversionPass> {
public:
  using RuntimeToFuncConversionPassBase<
      RuntimeToFuncConversionPass>::RuntimeToFuncConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult groupModelOps();

  mlir::LogicalResult convertOps();
};
} // namespace

void RuntimeToFuncConversionPass::runOnOperation() {
  if (mlir::failed(groupModelOps())) {
    return signalPassFailure();
  }

  if (mlir::failed(convertOps())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult RuntimeToFuncConversionPass::groupModelOps() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());

  llvm::SmallVector<ICModelBeginOp> icModelBeginOps;
  llvm::SmallVector<ICModelEndOp> icModelEndOps;
  llvm::SmallVector<DynamicModelBeginOp> dynamicModelBeginOps;
  llvm::SmallVector<DynamicModelEndOp> dynamicModelEndOps;

  for (auto &op : moduleOp.getOps()) {
    if (auto icModelBeginOp = mlir::dyn_cast<ICModelBeginOp>(op)) {
      icModelBeginOps.push_back(icModelBeginOp);
      continue;
    }

    if (auto icModelEndOp = mlir::dyn_cast<ICModelEndOp>(op)) {
      icModelEndOps.push_back(icModelEndOp);
      continue;
    }

    if (auto dynamicModelBeginOp = mlir::dyn_cast<DynamicModelBeginOp>(op)) {
      dynamicModelBeginOps.push_back(dynamicModelBeginOp);
      continue;
    }

    if (auto dynamicModelEndOp = mlir::dyn_cast<DynamicModelEndOp>(op)) {
      dynamicModelEndOps.push_back(dynamicModelEndOp);
      continue;
    }
  }

  if (icModelBeginOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<ICModelBeginOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (ICModelBeginOp op : icModelBeginOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (icModelEndOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<ICModelEndOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (ICModelEndOp op : icModelEndOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (dynamicModelBeginOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<DynamicModelBeginOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (DynamicModelBeginOp op : dynamicModelBeginOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (dynamicModelEndOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<DynamicModelEndOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (DynamicModelEndOp op : dynamicModelEndOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  return mlir::success();
}

mlir::LogicalResult RuntimeToFuncConversionPass::convertOps() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<VariableGetterOp, InitFunctionOp, DeinitFunctionOp,
                      ICModelBeginOp, ICModelEndOp, DynamicModelBeginOp,
                      DynamicModelEndOp, EquationFunctionOp, ReturnOp>();

  target.addDynamicallyLegalOp<FunctionOp>(
      [](FunctionOp op) { return op.isDeclaration(); });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<VariableGetterOpLowering, InitFunctionOpLowering,
                  DeinitFunctionOpLowering, ICModelBeginOpLowering,
                  ICModelEndOpLowering, DynamicModelBeginOpLowering,
                  DynamicModelEndOpLowering, EquationFunctionOpLowering,
                  FunctionOpLowering, ReturnOpLowering>(&getContext());

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

namespace mlir {
std::unique_ptr<mlir::Pass> createRuntimeToFuncConversionPass() {
  return std::make_unique<RuntimeToFuncConversionPass>();
}
} // namespace mlir
