#include "marco/Codegen/Conversion/BaseModelicaToTensor/BaseModelicaToTensor.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOTENSORCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
class BaseModelicaToTensorConversionPass
    : public mlir::impl::BaseModelicaToTensorConversionPassBase<
          BaseModelicaToTensorConversionPass> {
public:
  using BaseModelicaToTensorConversionPassBase<
      BaseModelicaToTensorConversionPass>::
      BaseModelicaToTensorConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

void BaseModelicaToTensorConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

namespace {
struct TensorFromElementsOpLowering
    : public mlir::OpConversionPattern<TensorFromElementsOp> {
  using mlir::OpConversionPattern<TensorFromElementsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorFromElementsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto resultTensorType = mlir::cast<mlir::TensorType>(resultType);
    auto resultElementType = resultTensorType.getElementType();

    llvm::SmallVector<mlir::Value, 10> operands;

    for (mlir::Value operand : adaptor.getValues()) {
      if (operand.getType() != resultElementType) {
        operand =
            rewriter.create<CastOp>(op.getLoc(), resultElementType, operand);
      }

      operands.push_back(operand);
    }

    rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
        op, resultTensorType, operands);

    return mlir::success();
  }
};

struct TensorBroadcastOpLowering
    : public mlir::OpConversionPattern<TensorBroadcastOp> {
  using mlir::OpConversionPattern<TensorBroadcastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorBroadcastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto resultTensorType = mlir::cast<mlir::TensorType>(resultType);
    auto resultElementType = resultTensorType.getElementType();

    mlir::Value operand = adaptor.getValue();

    if (operand.getType() != resultElementType) {
      operand =
          rewriter.create<CastOp>(op.getLoc(), resultElementType, operand);
    }

    rewriter.replaceOpWithNewOp<mlir::tensor::SplatOp>(op, resultTensorType,
                                                       operand);

    return mlir::success();
  }
};

struct TensorViewOpLowering : public mlir::OpConversionPattern<TensorViewOp> {
  using mlir::OpConversionPattern<TensorViewOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    llvm::SmallVector<mlir::OpFoldResult, 10> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 10> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 10> strides;

    llvm::SmallVector<int64_t, 10> resultShape;

    for (mlir::Value subscript : adaptor.getSubscriptions()) {
      if (mlir::isa<RangeType>(subscript.getType())) {
        mlir::Value begin = rewriter.create<RangeBeginOp>(loc, subscript);
        mlir::Value size = rewriter.create<RangeSizeOp>(loc, subscript);
        mlir::Value step = rewriter.create<RangeStepOp>(loc, subscript);

        if (!mlir::isa<mlir::IndexType>(begin.getType())) {
          begin = rewriter.create<CastOp>(begin.getLoc(),
                                          rewriter.getIndexType(), begin);
        }

        if (!mlir::isa<mlir::IndexType>(size.getType())) {
          size = rewriter.create<CastOp>(size.getLoc(), rewriter.getIndexType(),
                                         size);
        }

        if (!mlir::isa<mlir::IndexType>(step.getType())) {
          step = rewriter.create<CastOp>(step.getLoc(), rewriter.getIndexType(),
                                         step);
        }

        offsets.push_back(begin);
        sizes.push_back(size);
        strides.push_back(step);
        resultShape.push_back(mlir::ShapedType::kDynamic);
      } else {
        offsets.push_back(subscript);
        sizes.push_back(rewriter.getI64IntegerAttr(1));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
    }

    auto numOfSubscripts =
        static_cast<int64_t>(adaptor.getSubscriptions().size());

    auto sourceTensorType =
        mlir::cast<mlir::TensorType>(adaptor.getSource().getType());

    int64_t sourceRank = sourceTensorType.getRank();

    for (int64_t i = numOfSubscripts; i < sourceRank; ++i) {
      offsets.push_back(rewriter.getI64IntegerAttr(0));
      int64_t sourceDimension = sourceTensorType.getDimSize(i);
      resultShape.push_back(sourceDimension);

      if (sourceDimension == mlir::ShapedType::kDynamic) {
        mlir::Value dimensionSize =
            rewriter.create<mlir::tensor::DimOp>(loc, adaptor.getSource(), i);

        sizes.push_back(dimensionSize);
      } else {
        sizes.push_back(rewriter.getI64IntegerAttr(sourceDimension));
      }

      strides.push_back(rewriter.getI64IntegerAttr(1));
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        mlir::cast<mlir::TensorType>(requestedResultType);

    auto resultType = requestedResultTensorType.clone(
        resultShape, requestedResultTensorType.getElementType());

    mlir::Value result = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, resultType, adaptor.getSource(), offsets, sizes, strides);

    if (result.getType() != requestedResultType) {
      result = rewriter.create<mlir::tensor::CastOp>(
          result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TensorExtractOpLowering
    : public mlir::OpConversionPattern<TensorExtractOp> {
  using mlir::OpConversionPattern<TensorExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value, 10> indices;

    for (mlir::Value index : adaptor.getIndices()) {
      if (!mlir::isa<mlir::IndexType>(index.getType())) {
        index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), index);
      }

      indices.push_back(index);
    }

    mlir::Value result = rewriter.create<mlir::tensor::ExtractOp>(
        loc, adaptor.getTensor(), indices);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result = rewriter.create<CastOp>(loc, requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TensorInsertOpLowering
    : public mlir::OpConversionPattern<TensorInsertOp> {
  using mlir::OpConversionPattern<TensorInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value tensor = adaptor.getDestination();
    auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());
    mlir::Type elementType = tensorType.getElementType();

    mlir::Value value = adaptor.getValue();

    if (value.getType() != elementType) {
      value = rewriter.create<CastOp>(loc, elementType, value);
    }

    llvm::SmallVector<mlir::Value, 10> indices;

    for (mlir::Value index : adaptor.getIndices()) {
      if (!mlir::isa<mlir::IndexType>(index.getType())) {
        index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), index);
      }

      indices.push_back(index);
    }

    mlir::Value result =
        rewriter.create<mlir::tensor::InsertOp>(loc, value, tensor, indices);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result = rewriter.create<CastOp>(loc, requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct TensorInsertSliceOpLowering
    : public mlir::OpConversionPattern<TensorInsertSliceOp> {
  using mlir::OpConversionPattern<TensorInsertSliceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorInsertSliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value source = adaptor.getValue();

    mlir::Value destination = adaptor.getDestination();

    auto destinationTensorType =
        mlir::cast<mlir::TensorType>(destination.getType());

    auto subscriptions = adaptor.getSubscriptions();
    size_t numOfSubscriptions = subscriptions.size();
    size_t subscriptionsIndex = 0;

    int64_t sourceDimension = 0;
    int64_t destinationRank = destinationTensorType.getRank();

    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;

    // Utility function to get a known dimension or create the operation to
    // obtain it at runtime.
    auto getDimSizeFn = [&](mlir::Value tensor,
                            int64_t dim) -> mlir::OpFoldResult {
      auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());
      assert(dim < tensorType.getRank());

      if (auto dimSize = tensorType.getDimSize(dim);
          dimSize != mlir::ShapedType::kDynamic) {
        return rewriter.getI64IntegerAttr(dimSize);
      }

      auto dimOp = rewriter.create<mlir::tensor::DimOp>(loc, tensor, dim);

      return dimOp.getResult();
    };

    // The source may have a rank smaller than the destination, so we iterate
    // on the destination rank.
    for (int64_t destinationDim = 0; destinationDim < destinationRank;
         ++destinationDim) {
      if (subscriptionsIndex < numOfSubscriptions) {
        mlir::Value subscription = subscriptions[subscriptionsIndex];

        if (mlir::isa<RangeType>(subscription.getType())) {
          // The offset is either the begin or the end of the range,
          // depending on the step value.
          // The size is given by the source dimension size.
          // The stride is given by the step.
          assert(sourceDimension <
                 mlir::cast<mlir::TensorType>(source.getType()).getRank());

          mlir::Value beginValue =
              rewriter.create<RangeBeginOp>(loc, subscription);

          mlir::Value endValue = rewriter.create<RangeEndOp>(loc, subscription);

          mlir::Value step = rewriter.create<RangeStepOp>(loc, subscription);

          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
              loc, rewriter.getIndexAttr(0));

          mlir::Value nonNegative = rewriter.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::sge, step, zero);

          mlir::Value offset = rewriter.create<mlir::arith::SelectOp>(
              loc, nonNegative, beginValue, endValue);

          offsets.push_back(offset);
          sizes.push_back(getDimSizeFn(source, sourceDimension++));
          strides.push_back(step);
        } else {
          // Use the subscription for reducing the rank of the destination
          // and add additional unitary dimensions to the source.
          offsets.push_back(subscription);
          sizes.push_back(rewriter.getI64IntegerAttr(1));
          strides.push_back(rewriter.getI64IntegerAttr(1));
        }

        ++subscriptionsIndex;
      } else {
        // No more subscriptions available.
        // The remaining dimensions are copied from the source into the
        // destination.
        assert(sourceDimension <
               mlir::cast<mlir::TensorType>(source.getType()).getRank());

        offsets.push_back(rewriter.getI64IntegerAttr(0));
        sizes.push_back(getDimSizeFn(source, sourceDimension++));
        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
    }

    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        op, source, destination, offsets, sizes, strides);

    return mlir::success();
  }
};

struct NDimsOpLowering : public mlir::OpConversionPattern<NDimsOp> {
  using mlir::OpConversionPattern<NDimsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NDimsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto tensorType =
        mlir::cast<mlir::TensorType>(adaptor.getArray().getType());

    mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIndexAttr(tensorType.getRank()));

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result = rewriter.create<CastOp>(loc, requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SizeOpDimensionLowering : public mlir::OpConversionPattern<SizeOp> {
  using mlir::OpConversionPattern<SizeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value tensor = adaptor.getArray();

    if (!op.hasDimension()) {
      return rewriter.notifyMatchFailure(op, "No index specified");
    }

    mlir::Value index = op.getDimension();

    if (!mlir::isa<mlir::IndexType>(index.getType())) {
      index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), index);
    }

    mlir::Value result =
        rewriter.create<mlir::tensor::DimOp>(loc, tensor, index);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result = rewriter.create<CastOp>(loc, requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SizeOpArrayLowering : public mlir::OpConversionPattern<SizeOp> {
  using mlir::OpConversionPattern<SizeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SizeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value tensor = adaptor.getArray();

    if (op.hasDimension()) {
      return rewriter.notifyMatchFailure(op, "Index specified");
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        mlir::cast<mlir::TensorType>(requestedResultType);

    mlir::Type requestedResultElementType =
        requestedResultTensorType.getElementType();

    llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

    if (requestedResultTensorType.getDimSize(0) == mlir::ShapedType::kDynamic) {
      dynamicDimensions.push_back(
          rewriter.create<mlir::tensor::RankOp>(loc, tensor));
    }

    mlir::Value result = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicDimensions);

    for (int64_t dim = 0, rank = requestedResultTensorType.getRank();
         dim < rank; ++dim) {
      mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(dim));

      mlir::Value size =
          rewriter.create<mlir::tensor::DimOp>(loc, tensor, index);

      if (size.getType() != requestedResultElementType) {
        size = rewriter.create<CastOp>(loc, requestedResultElementType, size);
      }

      result =
          rewriter.create<mlir::tensor::InsertOp>(loc, size, result, index);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct FillOpLowering : public mlir::OpRewritePattern<FillOp> {
  using mlir::OpRewritePattern<FillOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(FillOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TensorBroadcastOp>(op, op.getResult().getType(),
                                                   op.getValue());

    return mlir::success();
  }
};

struct DimOpLowering : public mlir::OpConversionPattern<mlir::tensor::DimOp> {
  using mlir::OpConversionPattern<mlir::tensor::DimOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (llvm::all_of(llvm::zip(op.getOperands(), adaptor.getOperands()),
                     [](const auto &operands) {
                       return std::get<0>(operands).getType() ==
                              std::get<1>(operands).getType();
                     })) {
      return rewriter.notifyMatchFailure(op, "Already legal operand types");
    }

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, adaptor.getSource(),
                                                     adaptor.getIndex());

    return mlir::success();
  }
};
} // namespace

mlir::LogicalResult BaseModelicaToTensorConversionPass::convertOperations() {
  auto moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();

  target.addIllegalOp<TensorFromElementsOp, TensorBroadcastOp, TensorViewOp,
                      TensorExtractOp, TensorInsertOp, TensorInsertSliceOp,
                      NDimsOp, SizeOp, FillOp>();

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::DataLayout dataLayout(moduleOp);
  TypeConverter typeConverter(&getContext(), dataLayout);

  mlir::RewritePatternSet patterns(&getContext());

  populateBaseModelicaToTensorConversionPatterns(patterns, &getContext(),
                                                 typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateBaseModelicaToTensorConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns
      .insert<TensorFromElementsOpLowering, TensorBroadcastOpLowering,
              TensorViewOpLowering, TensorExtractOpLowering,
              TensorInsertOpLowering, TensorInsertSliceOpLowering,
              NDimsOpLowering, SizeOpDimensionLowering, SizeOpArrayLowering>(
          typeConverter, context);

  patterns.insert<DimOpLowering>(typeConverter, context);
  patterns.insert<FillOpLowering>(context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToTensorConversionPass() {
  return std::make_unique<BaseModelicaToTensorConversionPass>();
}
} // namespace mlir
