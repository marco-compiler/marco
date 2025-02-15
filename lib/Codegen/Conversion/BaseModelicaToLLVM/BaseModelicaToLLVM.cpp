#include "marco/Codegen/Conversion/BaseModelicaToLLVM/BaseModelicaToLLVM.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;
using namespace ::marco::codegen;

namespace {
template <typename Op>
class BaseModelicaOpConversion : public mlir::ConvertOpToLLVMPattern<Op> {
public:
  BaseModelicaOpConversion(mlir::LLVMTypeConverter &typeConverter,
                           mlir::SymbolTableCollection &symbolTableCollection)
      : mlir::ConvertOpToLLVMPattern<Op>(typeConverter),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::LLVM::LLVMFunctionType functionType) const {
    auto funcOp = symbolTableCollection->lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
        moduleOp, builder.getStringAttr(name));

    if (funcOp) {
      return funcOp;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto newFuncOp =
        builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, functionType);

    symbolTableCollection->getSymbolTable(moduleOp).insert(newFuncOp);
    return newFuncOp;
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Type> argTypes) const {
    auto functionType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes);

    return getOrDeclareFunction(builder, moduleOp, loc, name, functionType);
  }

  mlir::LLVM::LLVMFuncOp
  getOrDeclareFunction(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                       mlir::Location loc, llvm::StringRef name,
                       mlir::Type resultType,
                       llvm::ArrayRef<mlir::Value> args) const {
    llvm::SmallVector<mlir::Type> argTypes;

    for (mlir::Value arg : args) {
      argTypes.push_back(arg.getType());
    }

    return getOrDeclareFunction(builder, moduleOp, loc, name, resultType,
                                argTypes);
  }

protected:
  mlir::SymbolTableCollection *symbolTableCollection;
};

class PackageOpPattern : public mlir::OpRewritePattern<PackageOp> {
public:
  using mlir::OpRewritePattern<PackageOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(PackageOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class RecordOpPattern : public mlir::OpRewritePattern<RecordOp> {
public:
  using mlir::OpRewritePattern<RecordOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RecordOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConstantOpRangeLowering : public BaseModelicaOpConversion<ConstantOp> {
public:
  using BaseModelicaOpConversion<ConstantOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (!op.getResult().getType().isa<RangeType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible attribute");
    }

    auto structType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<mlir::LLVM::LLVMStructType>();

    mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

    if (auto rangeAttr = op.getValue().dyn_cast<IntegerRangeAttr>()) {
      auto lowerBoundAttr = rewriter.getIntegerAttr(structType.getBody()[0],
                                                    rangeAttr.getLowerBound());

      auto upperBoundAttr = rewriter.getIntegerAttr(structType.getBody()[1],
                                                    rangeAttr.getUpperBound());

      auto stepAttr =
          rewriter.getIntegerAttr(structType.getBody()[2], rangeAttr.getStep());

      mlir::Value lowerBound =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

      mlir::Value upperBound =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

      mlir::Value step = rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, structType, result, lowerBound, 0);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, structType, result, upperBound, 1);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType,
                                                          result, step, 2);

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (auto rangeAttr = op.getValue().dyn_cast<RealRangeAttr>()) {
      auto lowerBoundAttr = rewriter.getFloatAttr(
          structType.getBody()[0], rangeAttr.getLowerBound().convertToDouble());

      auto upperBoundAttr = rewriter.getFloatAttr(
          structType.getBody()[1], rangeAttr.getUpperBound().convertToDouble());

      auto stepAttr = rewriter.getFloatAttr(
          structType.getBody()[2], rangeAttr.getStep().convertToDouble());

      mlir::Value lowerBound =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

      mlir::Value upperBound =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

      mlir::Value step = rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, structType, result, lowerBound, 0);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, structType, result, upperBound, 1);

      result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType,
                                                          result, step, 2);

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

class RangeOpLowering : public BaseModelicaOpConversion<RangeOp> {
public:
  using BaseModelicaOpConversion<RangeOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(RangeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    auto structType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<mlir::LLVM::LLVMStructType>();

    mlir::Value lowerBound = adaptor.getLowerBound();

    if (mlir::Type requiredType = structType.getBody()[0];
        lowerBound.getType() != requiredType) {
      lowerBound = rewriter.create<CastOp>(loc, requiredType, lowerBound);
    }

    mlir::Value upperBound = adaptor.getUpperBound();

    if (mlir::Type requiredType = structType.getBody()[1];
        upperBound.getType() != requiredType) {
      upperBound = rewriter.create<CastOp>(loc, requiredType, upperBound);
    }

    mlir::Value step = adaptor.getStep();

    if (mlir::Type requiredType = structType.getBody()[2];
        step.getType() != requiredType) {
      step = rewriter.create<CastOp>(loc, requiredType, step);
    }

    mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType, result,
                                                        lowerBound, 0);

    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType, result,
                                                        upperBound, 1);

    result = rewriter.create<mlir::LLVM::InsertValueOp>(loc, structType, result,
                                                        step, 2);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct RangeBeginOpLowering : public BaseModelicaOpConversion<RangeBeginOp> {
  using BaseModelicaOpConversion<RangeBeginOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(RangeBeginOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getRange(), 0);

    return mlir::success();
  }
};

struct RangeEndOpLowering : public BaseModelicaOpConversion<RangeEndOp> {
  using BaseModelicaOpConversion<RangeEndOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(RangeEndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getRange(), 1);

    return mlir::success();
  }
};

struct RangeStepOpLowering : public BaseModelicaOpConversion<RangeStepOp> {
  using BaseModelicaOpConversion<RangeStepOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(RangeStepOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getRange(), 2);

    return mlir::success();
  }
};

struct PoolVariableGetOpLowering
    : public BaseModelicaOpConversion<PoolVariableGetOp> {
  using BaseModelicaOpConversion<PoolVariableGetOp>::BaseModelicaOpConversion;

  mlir::LogicalResult
  matchAndRewrite(PoolVariableGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getType().isa<mlir::MemRefType>()) {
      return rewriter.notifyMatchFailure(op, "Incompatible type");
    }

    mlir::Location loc = op.getLoc();
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    RuntimeFunctionsMangling mangling;

    llvm::SmallVector<mlir::Value, 2> args;
    llvm::SmallVector<std::string, 2> mangledArgsTypes;

    // Memory pool identifier.
    args.push_back(adaptor.getPool());
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    // Buffer identifier.
    args.push_back(adaptor.getId());
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    // Create the call to the runtime library.
    auto resultType = mlir::LLVM::LLVMPointerType::get(op.getContext());
    auto mangledResultType = mangling.getVoidPointerType();

    auto functionName = mangling.getMangledFunction(
        "memoryPoolGet", mangledResultType, mangledArgsTypes);

    auto funcOp = getOrDeclareFunction(rewriter, moduleOp, loc, functionName,
                                       resultType, args);

    auto callOp = rewriter.create<mlir::LLVM::CallOp>(loc, funcOp, args);

    mlir::Value ptr = callOp.getResult();
    auto memRefType = op.getType().cast<mlir::MemRefType>();

    ptr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptr.getType(), memRefType.getElementType(), ptr,
        llvm::ArrayRef<mlir::LLVM::GEPArg>(0));

    auto memRefDescriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memRefType, ptr, ptr);

    rewriter.replaceOp(op, {memRefDescriptor});
    return mlir::success();
  }
};
} // namespace

namespace {
class BaseModelicaToLLVMConversionPass
    : public mlir::impl::BaseModelicaToLLVMConversionPassBase<
          BaseModelicaToLLVMConversionPass> {
public:
  using BaseModelicaToLLVMConversionPassBase<
      BaseModelicaToLLVMConversionPass>::BaseModelicaToLLVMConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

void BaseModelicaToLLVMConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToLLVMConversionPass::convertOperations() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();

  target.addDynamicallyLegalOp<ConstantOp>(
      [](ConstantOp op) { return !op.getResult().getType().isa<RangeType>(); });

  target.addIllegalOp<PackageOp, RecordOp>();

  target.addIllegalOp<RangeOp, RangeBeginOp, RangeEndOp, RangeStepOp>();

  target.addIllegalOp<PoolVariableGetOp>();

  target.addDynamicallyLegalOp<PoolVariableGetOp>([](PoolVariableGetOp op) {
    return !op.getType().isa<mlir::MemRefType>();
  });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::DataLayout dataLayout(moduleOp);
  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext(), dataLayout);

  LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);
  mlir::SymbolTableCollection symbolTableCollection;

  mlir::RewritePatternSet patterns(&getContext());
  populateBaseModelicaToLLVMConversionPatterns(patterns, typeConverter,
                                               symbolTableCollection);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateBaseModelicaToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  // Class operations.
  patterns.insert<PackageOpPattern, RecordOpPattern>(
      &typeConverter.getContext());

  // Range operations.
  patterns
      .insert<ConstantOpRangeLowering, RangeOpLowering, RangeBeginOpLowering,
              RangeEndOpLowering, RangeStepOpLowering>(typeConverter,
                                                       symbolTableCollection);

  // Variable operations.
  patterns.insert<PoolVariableGetOpLowering>(typeConverter,
                                             symbolTableCollection);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass() {
  return std::make_unique<BaseModelicaToLLVMConversionPass>();
}
} // namespace mlir
