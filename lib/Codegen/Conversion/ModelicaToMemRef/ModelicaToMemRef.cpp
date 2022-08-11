#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      ModelicaOpRewritePattern(mlir::MLIRContext* ctx, ModelicaToMemRefOptions options)
          : mlir::OpRewritePattern<Op>(ctx),
            options(std::move(options))
      {
      }

    protected:
      ModelicaToMemRefOptions options;
  };

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

  struct NDimsOpLowering : public ModelicaOpRewritePattern<NDimsOp>
  {
    using ModelicaOpRewritePattern<NDimsOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NDimsOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct SizeOpDimensionLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "No index specified");
      }

      mlir::Value index = op.getDimension();

      if (!index.getType().isa<mlir::IndexType>()) {
        index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), index);
      }

      mlir::Value result = rewriter.create<DimOp>(loc, op.getArray(), index);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct SizeOpArrayLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      if (op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "Index specified");
      }

      assert(op.getResult().getType().isa<ArrayType>());
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, llvm::None);

      // Iterate on each dimension
      mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zeroValue, rank, step);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());

        // Get the size of the current dimension
        mlir::Value dimensionSize = rewriter.create<SizeOp>(loc, resultArrayType.getElementType(), op.getArray(), loop.getInductionVar());

        // Store it into the result array
        rewriter.create<StoreOp>(loc, dimensionSize, result, loop.getInductionVar());
      }

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

        target.addIllegalOp<
            NDimsOp,
            SizeOp>();

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

    patterns.insert<
        NDimsOpLowering,
        SizeOpDimensionLowering,
        SizeOpArrayLowering>(context, options);
  }

  std::unique_ptr<mlir::Pass> createModelicaToMemRefPass(ModelicaToMemRefOptions options)
  {
    return std::make_unique<ModelicaToMemRefConversionPass>(options);
  }
}
