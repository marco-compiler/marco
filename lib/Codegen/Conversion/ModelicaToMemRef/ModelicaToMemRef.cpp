#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOMEMREFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      using mlir::OpRewritePattern<Op>::OpRewritePattern;
  };

  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op>
  {
    public:
      using mlir::OpConversionPattern<Op>::OpConversionPattern;
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

  struct AssignmentOpScalarCastPattern : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AssignmentOp op) const override
    {
      if (!isNumeric(op.getValue())) {
        return mlir::failure();
      }

      mlir::Type valueType = op.getValue().getType();
      mlir::Type elementType = op.getDestination().getType().cast<ArrayType>().getElementType();

      return mlir::LogicalResult::success(valueType != elementType);
    }

    void rewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type elementType = op.getDestination().getType().cast<ArrayType>().getElementType();
      mlir::Value value = rewriter.create<CastOp>(loc, elementType, op.getValue());

      rewriter.replaceOpWithNewOp<AssignmentOp>(op, op.getDestination(), value);
    }
  };

  struct AssignmentOpScalarLowering : public ModelicaOpConversionPattern<AssignmentOp>
  {
    using ModelicaOpConversionPattern<AssignmentOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AssignmentOp op) const override
    {
      if (!isNumeric(op.getValue())) {
        return mlir::failure();
      }

      mlir::Type valueType = op.getValue().getType();
      mlir::Type elementType = op.getDestination().getType().cast<ArrayType>().getElementType();

      return mlir::LogicalResult::success(valueType == elementType);
    }

    void rewrite(AssignmentOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, adaptor.getValue(), adaptor.getDestination());
    }
  };

  struct AssignmentOpArrayLowering : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.getValue().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source value is not an array");
      }

      mlir::Value destination = op.getDestination();

      assert(destination.getType().isa<ArrayType>());
      auto arrayType = destination.getType().cast<ArrayType>();

      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      llvm::SmallVector<mlir::Value, 3> lowerBounds(arrayType.getRank(), zero);
      llvm::SmallVector<mlir::Value, 3> upperBounds;
      llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

      for (unsigned int i = 0, e = arrayType.getRank(); i < e; ++i) {
        mlir::Value dim = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
        upperBounds.push_back(rewriter.create<DimOp>(loc, destination, dim));
      }

      // Create nested loops in order to iterate on each dimension of the array
      mlir::scf::buildLoopNest(
          rewriter, loc, lowerBounds, upperBounds, steps,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location, mlir::ValueRange position) {
            mlir::Value value = rewriter.create<LoadOp>(loc, op.getValue(), position);
            value = rewriter.create<CastOp>(value.getLoc(), op.getDestination().getType().cast<ArrayType>().getElementType(), value);
            rewriter.create<StoreOp>(loc, value, op.getDestination(), position);
          });

      rewriter.eraseOp(op);
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

      mlir::Value result = rewriter.create<mlir::memref::SubViewOp>(
          loc,
          resultType.cast<mlir::MemRefType>(),
          adaptor.getSource(),
          offsets, sizes, strides);

      rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(
          op,
          getTypeConverter()->convertType(op.getResult().getType()),
          result);

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

static void populateModelicaToMemRefPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::TypeConverter& typeConverter)
{
  patterns.insert<
      ArrayCastOpLowering>(typeConverter, context);

  patterns.insert<
      AssignmentOpScalarCastPattern,
      AssignmentOpArrayLowering>(context);

  patterns.insert<
      AssignmentOpScalarLowering>(typeConverter, context);

  patterns.insert<
      AllocaOpLowering,
      AllocOpLowering,
      FreeOpLowering,
      DimOpLowering,
      SubscriptionOpLowering,
      LoadOpLowering,
      StoreOpLowering>(typeConverter, context);

  patterns.insert<
      NDimsOpLowering,
      SizeOpDimensionLowering,
      SizeOpArrayLowering>(context);
}

namespace
{
  class ModelicaToMemRefConversionPass : public mlir::impl::ModelicaToMemRefConversionPassBase<ModelicaToMemRefConversionPass>
  {
    public:
      using ModelicaToMemRefConversionPassBase::ModelicaToMemRefConversionPassBase;

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

        target.addIllegalOp<
            ArrayCastOp,
            AssignmentOp>();

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

        TypeConverter typeConverter(bitWidth);

        typeConverter.addConversion([](mlir::IndexType type) {
          return type;
        });

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToMemRefPatterns(patterns, &getContext(), typeConverter);

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToMemRefConversionPass()
  {
    return std::make_unique<ModelicaToMemRefConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToMemRefConversionPass(const ModelicaToMemRefConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToMemRefConversionPass>(options);
  }
}
