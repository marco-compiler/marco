#include "marco/Codegen/Conversion/BaseModelicaToLLVM/BaseModelicaToLLVM.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/LLVMTypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_BASEMODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::bmodelica;

namespace
{
  class PackageOpPattern : public mlir::OpRewritePattern<PackageOp>
  {
    public:
      using mlir::OpRewritePattern<PackageOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          PackageOp op, mlir::PatternRewriter& rewriter) const override
      {
        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class RecordOpPattern : public mlir::OpRewritePattern<RecordOp>
  {
    public:
      using mlir::OpRewritePattern<RecordOp>::OpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          RecordOp op, mlir::PatternRewriter& rewriter) const override
      {
        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class ConstantOpRangeLowering
      : public mlir::ConvertOpToLLVMPattern<ConstantOp>
  {
    public:
      using mlir::ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;

      mlir::LogicalResult matchAndRewrite(
          ConstantOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        if (!op.getResult().getType().isa<RangeType>()) {
          return rewriter.notifyMatchFailure(op, "Incompatible attribute");
        }

        auto structType =
            getTypeConverter()->convertType(op.getResult().getType())
                .cast<mlir::LLVM::LLVMStructType>();

        mlir::Value result =
            rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

        if (auto rangeAttr = op.getValue().dyn_cast<IntegerRangeAttr>()) {
          auto lowerBoundAttr = rewriter.getIntegerAttr(
              structType.getBody()[0], rangeAttr.getLowerBound());

          auto upperBoundAttr = rewriter.getIntegerAttr(
              structType.getBody()[1], rangeAttr.getUpperBound());

          auto stepAttr = rewriter.getIntegerAttr(
              structType.getBody()[2], rangeAttr.getStep());

          mlir::Value lowerBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

          mlir::Value upperBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

          mlir::Value step =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, lowerBound, 0);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, upperBound, 1);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, step, 2);

          rewriter.replaceOp(op, result);
          return mlir::success();
        }

        if (auto rangeAttr = op.getValue().dyn_cast<RealRangeAttr>()) {
          auto lowerBoundAttr = rewriter.getFloatAttr(
              structType.getBody()[0],
              rangeAttr.getLowerBound().convertToDouble());

          auto upperBoundAttr = rewriter.getFloatAttr(
              structType.getBody()[1],
              rangeAttr.getUpperBound().convertToDouble());

          auto stepAttr = rewriter.getFloatAttr(
              structType.getBody()[2],
              rangeAttr.getStep().convertToDouble());

          mlir::Value lowerBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

          mlir::Value upperBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

          mlir::Value step =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, lowerBound, 0);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, upperBound, 1);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, step, 2);

          rewriter.replaceOp(op, result);
          return mlir::success();
        }

        return mlir::failure();
      }
  };

  class RangeOpLowering
      : public mlir::ConvertOpToLLVMPattern<RangeOp>
  {
    public:
      using mlir::ConvertOpToLLVMPattern<RangeOp>::ConvertOpToLLVMPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto structType =
            getTypeConverter()->convertType(op.getResult().getType())
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

        mlir::Value result =
            rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, lowerBound, 0);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, upperBound, 1);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, step, 2);

        rewriter.replaceOp(op, result);
        return mlir::success();
      }
  };

  struct RangeBeginOpLowering
      : public mlir::ConvertOpToLLVMPattern<RangeBeginOp>
  {
      using mlir::ConvertOpToLLVMPattern<RangeBeginOp>::ConvertOpToLLVMPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeBeginOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 0);

        return mlir::success();
      }
  };

  struct RangeEndOpLowering
      : public mlir::ConvertOpToLLVMPattern<RangeEndOp>
  {
      using mlir::ConvertOpToLLVMPattern<RangeEndOp>::ConvertOpToLLVMPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeEndOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 1);

        return mlir::success();
      }
  };

  struct RangeStepOpLowering : public mlir::ConvertOpToLLVMPattern<RangeStepOp>
  {
      using mlir::ConvertOpToLLVMPattern<RangeStepOp>::ConvertOpToLLVMPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeStepOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 2);

        return mlir::success();
      }
  };
}

namespace
{
  class BaseModelicaToLLVMConversionPass
      : public mlir::impl::BaseModelicaToLLVMConversionPassBase<
            BaseModelicaToLLVMConversionPass>
  {
    public:
      using BaseModelicaToLLVMConversionPassBase<
          BaseModelicaToLLVMConversionPass>
          ::BaseModelicaToLLVMConversionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult convertOperations();
  };
}

void BaseModelicaToLLVMConversionPass::runOnOperation()
{
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToLLVMConversionPass::convertOperations()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();

  target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
    return !op.getResult().getType().isa<RangeType>();
  });

  target.addIllegalOp<
      PackageOp,
      RecordOp>();

  target.addIllegalOp<
      RangeOp,
      RangeBeginOp,
      RangeEndOp,
      RangeStepOp>();

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions);

  mlir::RewritePatternSet patterns(&getContext());
  populateBaseModelicaToLLVMConversionPatterns(patterns, typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace
{
  struct BaseModelicaToLLVMDialectInterface
      : public mlir::ConvertToLLVMPatternInterface
  {
    using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

    void loadDependentDialects(mlir::MLIRContext* context) const final
    {
      context->loadDialect<mlir::LLVM::LLVMDialect>();
    }

    void populateConvertToLLVMConversionPatterns(
        mlir::ConversionTarget& target,
        mlir::LLVMTypeConverter& typeConverter,
        mlir::RewritePatternSet &patterns) const final
    {
      populateBaseModelicaToLLVMConversionPatterns(patterns, typeConverter);
    }
  };
}

namespace mlir
{
  void populateBaseModelicaToLLVMConversionPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::LLVMTypeConverter& typeConverter)
  {
    // Class operations.
    patterns.insert<
        PackageOpPattern,
        RecordOpPattern>(&typeConverter.getContext());

    // Range operations.
    patterns.insert<
        ConstantOpRangeLowering,
        RangeOpLowering,
        RangeBeginOpLowering,
        RangeEndOpLowering,
        RangeStepOpLowering>(typeConverter);
  }

  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass()
  {
    return std::make_unique<BaseModelicaToLLVMConversionPass>();
  }

  void registerConvertBaseModelicaToLLVMInterface(mlir::DialectRegistry& registry)
  {
    registry.addExtension(
        +[](mlir::MLIRContext* context, BaseModelicaDialect* dialect) {
          dialect->addInterfaces<BaseModelicaToLLVMDialectInterface>();
        });
  }
}
