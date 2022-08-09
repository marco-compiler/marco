#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Codegen/ArrayDescriptor.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op>
  {
    public:
      ModelicaOpConversionPattern(mlir::MLIRContext* context, mlir::TypeConverter& typeConverter, ModelicaToMemRefOptions options)
          : mlir::OpConversionPattern<Op>(typeConverter, context),
            options(std::move(options))
      {
      }

    protected:
      ModelicaToMemRefOptions options;
  };
}

namespace
{
  struct ArrayCastOpLowering : public ModelicaOpConversionPattern<ArrayCastOp>
  {
    using ModelicaOpConversionPattern<ArrayCastOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ArrayCastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, resultType, adaptor.getSource());
      return mlir::success();
    }
  };

  class AllocaOpLowering : public ModelicaOpConversionPattern<AllocaOp>
  {
    using ModelicaOpConversionPattern<AllocaOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AllocaOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto memRefType = getTypeConverter()->convertType(op.getResult().getType()).cast<mlir::MemRefType>();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(op, memRefType, adaptor.getDynamicSizes());
      return mlir::success();
    }
  };

  class AllocOpLowering : public ModelicaOpConversionPattern<AllocOp>
  {
    using ModelicaOpConversionPattern<AllocOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AllocOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto memRefType = getTypeConverter()->convertType(op.getResult().getType()).cast<mlir::MemRefType>();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(op, memRefType, adaptor.getDynamicSizes());
      return mlir::success();
    }
  };

  class FreeOpLowering : public ModelicaOpConversionPattern<FreeOp>
  {
    using ModelicaOpConversionPattern<FreeOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(FreeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::memref::DeallocOp>(op, adaptor.getArray());
      return mlir::success();
    }
  };

  class DimOpLowering : public ModelicaOpConversionPattern<DimOp>
  {
    using ModelicaOpConversionPattern<DimOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DimOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::memref::DimOp>(op, adaptor.getArray(), adaptor.getDimension());
      return mlir::success();
    }
  };

  class SubscriptionOpLowering : public ModelicaOpConversionPattern<SubscriptionOp>
  {
    using ModelicaOpConversionPattern<SubscriptionOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SubscriptionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::OpFoldResult, 3> offsets;
      llvm::SmallVector<mlir::OpFoldResult, 3> sizes;
      llvm::SmallVector<mlir::OpFoldResult, 3> strides;

      auto sourceRank = op.getSource().getType().cast<ArrayType>().getRank();
      auto resultRank = op.getResult().getType().cast<ArrayType>().getRank();
      auto subscriptions = sourceRank - resultRank;

      for (unsigned int i = 0; i < sourceRank; ++i) {
        if (i < subscriptions) {
          offsets.push_back(adaptor.getIndices()[i]);
          sizes.push_back(rewriter.getI64IntegerAttr(1));
        } else {
          offsets.push_back(rewriter.getI64IntegerAttr(0));

          auto sourceDimension = adaptor.getSource().getType().cast<mlir::MemRefType>().getShape()[i];

          if (sourceDimension == mlir::ShapedType::kDynamicSize) {
            mlir::Value size = rewriter.create<mlir::memref::DimOp>(loc, adaptor.getSource(), i);
            sizes.push_back(size);
          } else {
            sizes.push_back(rewriter.getI64IntegerAttr(sourceDimension));
          }
        }

        strides.push_back(rewriter.getI64IntegerAttr(1));
      }

      auto resultType = mlir::memref::SubViewOp::inferRankReducedResultType(
          op.getResult().getType().cast<ArrayType>().getShape(),
          adaptor.getSource().getType().cast<mlir::MemRefType>(),
          offsets, sizes, strides);

      rewriter.replaceOpWithNewOp<mlir::memref::SubViewOp>(
          op,
          resultType.cast<mlir::MemRefType>(),
          adaptor.getSource(),
          offsets, sizes, strides);

      return mlir::success();
    }
  };

  class LoadOpLowering : public ModelicaOpConversionPattern<LoadOp>
  {
    using ModelicaOpConversionPattern<LoadOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getArray(), adaptor.getIndices());
      return mlir::success();
    }
  };

  class StoreOpLowering : public ModelicaOpConversionPattern<StoreOp>
  {
    using ModelicaOpConversionPattern<StoreOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(), adaptor.getArray(), adaptor.getIndices());
      return mlir::success();
    }
  };
}

namespace
{
  class ModelicaToMemRefConversionPass : public ModelicaToMemRefBase<ModelicaToMemRefConversionPass>
  {
    public:
      ModelicaToMemRefConversionPass(ModelicaToMemRefOptions options)
          : options(std::move(options))
      {
      }

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<ArrayCastOp>();

        target.addIllegalOp<
            AllocaOp,
            AllocOp,
            FreeOp,
            DimOp,
            SubscriptionOp,
            LoadOp,
            StoreOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        TypeConverter typeConverter(options.bitWidth);

        typeConverter.addConversion([](mlir::IndexType type) {
          return type;
        });

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToMemRefPatterns(patterns, &getContext(), typeConverter, options);

        return applyPartialConversion(module, target, std::move(patterns));
      }

    private:
      ModelicaToMemRefOptions options;
  };
}

namespace marco::codegen
{
  const ModelicaToMemRefOptions& ModelicaToMemRefOptions::getDefaultOptions()
  {
    static ModelicaToMemRefOptions options;
    return options;
  }

  void populateModelicaToMemRefPatterns(
      mlir::RewritePatternSet& patterns,
      mlir::MLIRContext* context,
      mlir::TypeConverter& typeConverter,
      ModelicaToMemRefOptions options)
  {
    patterns.insert<
        ArrayCastOpLowering>(context, typeConverter, options);

    patterns.insert<
        AllocaOpLowering,
        AllocOpLowering,
        FreeOpLowering,
        DimOpLowering,
        SubscriptionOpLowering,
        LoadOpLowering,
        StoreOpLowering>(context, typeConverter, options);
  }

  std::unique_ptr<mlir::Pass> createModelicaToMemRefPass(ModelicaToMemRefOptions options)
  {
    return std::make_unique<ModelicaToMemRefConversionPass>(options);
  }
}
