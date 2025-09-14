#include "marco/Codegen/Conversion/BaseModelicaToMemRef/BaseModelicaToMemRef.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOMEMREFCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
class BaseModelicaToMemRefConversionPass
    : public mlir::impl::BaseModelicaToMemRefConversionPassBase<
          BaseModelicaToMemRefConversionPass> {
public:
  using BaseModelicaToMemRefConversionPassBase<
      BaseModelicaToMemRefConversionPass>::
      BaseModelicaToMemRefConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

namespace {
/// Generic rewrite pattern that provides some utility functions.
template <typename Op>
class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;
};

/// Generic conversion pattern that provides some utility functions.
template <typename Op>
class ModelicaOpConversionPattern : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

protected:
  mlir::Attribute getDenseAttr(llvm::ArrayRef<int64_t> shape,
                               mlir::Attribute values) const {
    if (!values) {
      return {};
    }

    if (auto booleanArrayAttr =
            mlir::dyn_cast<DenseBooleanElementsAttr>(values)) {
      auto elementType = this->getTypeConverter()->convertType(
          mlir::cast<ArrayType>(booleanArrayAttr.getType()).getElementType());

      auto tensorType = mlir::RankedTensorType::get(shape, elementType);

      return mlir::DenseIntElementsAttr::get(tensorType,
                                             booleanArrayAttr.getValues());
    }

    if (auto integerArrayAttr =
            mlir::dyn_cast<DenseIntegerElementsAttr>(values)) {
      auto elementType = this->getTypeConverter()->convertType(
          mlir::cast<ArrayType>(integerArrayAttr.getType()).getElementType());

      auto tensorType = mlir::RankedTensorType::get(shape, elementType);
      return mlir::DenseIntElementsAttr::get(tensorType,
                                             integerArrayAttr.getValues());
    }

    if (auto realArrayAttr = mlir::dyn_cast<DenseRealElementsAttr>(values)) {
      llvm::SmallVector<double> casted;

      for (const auto &value : realArrayAttr.getValues()) {
        casted.push_back(value.convertToDouble());
      }

      auto elementType = this->getTypeConverter()->convertType(
          mlir::cast<ArrayType>(realArrayAttr.getType()).getElementType());

      auto tensorType = mlir::RankedTensorType::get(shape, elementType);
      return mlir::DenseFPElementsAttr::get(tensorType, casted);
    }

    return {};
  }
};
} // namespace

namespace {
struct ArrayCastOpLowering : public ModelicaOpConversionPattern<ArrayCastOp> {
  using ModelicaOpConversionPattern<ArrayCastOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(op, resultType,
                                                      adaptor.getSource());

    return mlir::success();
  }
};

class ConstantOpArrayLowering : public ModelicaOpConversionPattern<ConstantOp> {
public:
  ConstantOpArrayLowering(mlir::TypeConverter &typeConverter,
                          mlir::MLIRContext *context,
                          mlir::SymbolTableCollection &symbolTableCollection)
      : ModelicaOpConversionPattern<ConstantOp>(typeConverter, context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    auto memRefType = mlir::dyn_cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!memRefType) {
      return rewriter.notifyMatchFailure(op, "Incompatible result");
    }

    mlir::Attribute denseAttr = getDenseAttr(
        memRefType.getShape(), mlir::cast<mlir::Attribute>(op.getValue()));

    if (!denseAttr) {
      return rewriter.notifyMatchFailure(op, "Unknown attribute data type");
    }

    // Create the global constant.
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    auto globalOp =
        getOrCreateGlobalMemRef(rewriter, moduleOp, loc, memRefType, denseAttr);

    // Get the global constant.
    mlir::Value replacement = rewriter.create<mlir::memref::GetGlobalOp>(
        loc, memRefType, globalOp.getSymName());

    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

private:
  mlir::memref::GlobalOp
  getOrCreateGlobalMemRef(mlir::OpBuilder &builder, mlir::ModuleOp moduleOp,
                          mlir::Location loc, mlir::MemRefType memRefType,
                          mlir::Attribute denseAttr) const {
    for (mlir::memref::GlobalOp op :
         moduleOp.getOps<mlir::memref::GlobalOp>()) {
      auto initialValue = op.getInitialValue();

      if (!initialValue) {
        continue;
      }

      if (initialValue == denseAttr && op.getConstant()) {
        return op;
      }
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    mlir::SymbolTable &symbolTable =
        symbolTableCollection->getSymbolTable(moduleOp);

    auto globalOp = builder.create<mlir::memref::GlobalOp>(
        loc, "cst", builder.getStringAttr("private"), memRefType,
        mlir::cast<mlir::ElementsAttr>(denseAttr), true, nullptr);

    symbolTable.insert(globalOp);
    return globalOp;
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
};

class ArrayToTensorOpLowering
    : public ModelicaOpConversionPattern<ArrayToTensorOp> {
  using ModelicaOpConversionPattern<
      ArrayToTensorOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayToTensorOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensorType = getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<mlir::bufferization::ToTensorOp>(
        op, tensorType, adaptor.getArray(), true, true);

    return mlir::success();
  }
};

class TensorToArrayOpLowering
    : public ModelicaOpConversionPattern<TensorToArrayOp> {
  using ModelicaOpConversionPattern<
      TensorToArrayOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TensorToArrayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = getTypeConverter()->convertType(op.getResult().getType());

    rewriter.replaceOpWithNewOp<mlir::bufferization::ToBufferOp>(
        op, memrefType, adaptor.getTensor());

    return mlir::success();
  }
};

class GlobalVariableOpLowering
    : public ModelicaOpConversionPattern<GlobalVariableOp> {
public:
  GlobalVariableOpLowering(mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context,
                           mlir::SymbolTableCollection &symbolTableCollection)
      : ModelicaOpConversionPattern<GlobalVariableOp>(typeConverter, context),
        symbolTableCollection(&symbolTableCollection) {}

  mlir::LogicalResult
  matchAndRewrite(GlobalVariableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

    mlir::SymbolTable &symbolTable =
        symbolTableCollection->getSymbolTable(moduleOp);

    auto memRefType = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getType()));

    mlir::Attribute denseAttr = rewriter.getUnitAttr();

    if (auto initialValue = op.getInitialValue()) {
      denseAttr = getDenseAttr(memRefType.getShape(), *initialValue);
    }

    if (!denseAttr) {
      return rewriter.notifyMatchFailure(op, "Unknown attribute data type");
    }

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, op.getSymName(), rewriter.getStringAttr("private"), memRefType,
        denseAttr, false, nullptr);

    symbolTable.remove(op);
    return mlir::success();
  }

private:
  mlir::SymbolTableCollection *symbolTableCollection;
};

class GlobalVariableGetOpLowering
    : public ModelicaOpConversionPattern<GlobalVariableGetOp> {
  using ModelicaOpConversionPattern<
      GlobalVariableGetOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GlobalVariableGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memRefType = getTypeConverter()->convertType(op.getType());

    rewriter.replaceOpWithNewOp<mlir::memref::GetGlobalOp>(op, memRefType,
                                                           op.getVariable());

    return mlir::success();
  }
};

class PoolVariableGetOpLowering
    : public ModelicaOpConversionPattern<PoolVariableGetOp> {
  using ModelicaOpConversionPattern<
      PoolVariableGetOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(PoolVariableGetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<ArrayType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible type");
    }

    auto memRefType = getTypeConverter()->convertType(op.getType());

    rewriter.replaceOpWithNewOp<PoolVariableGetOp>(
        op, memRefType, adaptor.getPool(), adaptor.getId());

    return mlir::success();
  }
};

class AllocaOpLowering : public ModelicaOpConversionPattern<AllocaOp> {
  using ModelicaOpConversionPattern<AllocaOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memRefType = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(
        op, memRefType, adaptor.getDynamicSizes());

    return mlir::success();
  }
};

class AllocOpLowering : public ModelicaOpConversionPattern<AllocOp> {
  using ModelicaOpConversionPattern<AllocOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memRefType = mlir::cast<mlir::MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(
        op, memRefType, adaptor.getDynamicSizes());

    return mlir::success();
  }
};

class ArrayFromElementsOpLowering
    : public ModelicaOpRewritePattern<ArrayFromElementsOp> {
  using ModelicaOpRewritePattern<ArrayFromElementsOp>::ModelicaOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayFromElementsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    ArrayType arrayType = op.getArrayType();
    assert(arrayType.hasStaticShape());

    auto allocOp =
        rewriter.replaceOpWithNewOp<AllocOp>(op, arrayType, mlir::ValueRange());

    mlir::ValueRange values = op.getValues();
    size_t currentValue = 0;

    llvm::SmallVector<int64_t, 3> indices(arrayType.getRank(), 0);

    auto advanceIndices = [&]() -> bool {
      for (size_t i = 0, e = indices.size(); i < e; ++i) {
        size_t pos = e - i - 1;
        ++indices[pos];

        if (indices[pos] == arrayType.getDimSize(pos)) {
          indices[pos] = 0;
        } else {
          return true;
        }
      }

      return false;
    };

    llvm::SmallVector<mlir::Value, 3> indicesValues;

    do {
      for (int64_t index : indices) {
        indicesValues.push_back(rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(index)));
      }

      mlir::Value value = values[currentValue++];

      if (mlir::Type elementType = arrayType.getElementType();
          value.getType() != elementType) {
        value = rewriter.create<CastOp>(loc, elementType, value);
      }

      rewriter.create<StoreOp>(loc, value, allocOp.getResult(), indicesValues);

      indicesValues.clear();
    } while (advanceIndices());

    return mlir::success();
  }
};

class ArrayBroadcastOpLowering
    : public ModelicaOpRewritePattern<ArrayBroadcastOp> {
  using ModelicaOpRewritePattern<ArrayBroadcastOp>::ModelicaOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    ArrayType arrayType = op.getArrayType();

    auto allocOp = rewriter.replaceOpWithNewOp<AllocOp>(
        op, arrayType, op.getDynamicDimensions());

    rewriter.create<ArrayFillOp>(loc, allocOp.getResult(), op.getValue());
    return mlir::success();
  }
};

class FreeOpLowering : public ModelicaOpConversionPattern<FreeOp> {
  using ModelicaOpConversionPattern<FreeOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(FreeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::DeallocOp>(op,
                                                         adaptor.getArray());

    return mlir::success();
  }
};

class DimOpLowering : public ModelicaOpConversionPattern<DimOp> {
  using ModelicaOpConversionPattern<DimOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::DimOp>(op, adaptor.getArray(),
                                                     adaptor.getDimension());

    return mlir::success();
  }
};

class SubscriptionOpLowering
    : public ModelicaOpConversionPattern<SubscriptionOp> {
  using ModelicaOpConversionPattern<
      SubscriptionOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubscriptionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    auto sourceArrayType = mlir::cast<ArrayType>(op.getSource().getType());
    int64_t sourceRank = sourceArrayType.getRank();

    llvm::SmallVector<mlir::OpFoldResult, 3> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 3> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 3> strides;

    int64_t numOfSubscriptions = static_cast<int64_t>(op.getIndices().size());

    for (int64_t i = 0; i < sourceRank; ++i) {
      if (i < numOfSubscriptions) {
        mlir::Value subscription = op.getIndices()[i];

        if (mlir::isa<RangeType>(subscription.getType())) {
          mlir::Value lowerBound =
              rewriter.create<RangeBeginOp>(loc, subscription);

          mlir::Value step = rewriter.create<RangeStepOp>(loc, subscription);

          mlir::Value numOfElements =
              rewriter.create<RangeSizeOp>(loc, subscription);

          if (!mlir::isa<mlir::IndexType>(lowerBound.getType())) {
            lowerBound = rewriter.create<CastOp>(
                lowerBound.getLoc(), rewriter.getIndexType(), lowerBound);
          }

          if (!mlir::isa<mlir::IndexType>(step.getType())) {
            step = rewriter.create<CastOp>(step.getLoc(),
                                           rewriter.getIndexType(), step);
          }

          offsets.push_back(lowerBound);
          sizes.push_back(numOfElements);
          strides.push_back(step);
        } else {
          offsets.push_back(adaptor.getIndices()[i]);
          sizes.push_back(rewriter.getI64IntegerAttr(1));
          strides.push_back(rewriter.getI64IntegerAttr(1));
        }
      } else {
        offsets.push_back(rewriter.getI64IntegerAttr(0));
        int64_t sourceDimension = sourceArrayType.getDimSize(i);

        if (sourceDimension == mlir::ShapedType::kDynamic) {
          mlir::Value size =
              rewriter.create<mlir::memref::DimOp>(loc, adaptor.getSource(), i);

          sizes.push_back(size);
        } else {
          sizes.push_back(rewriter.getI64IntegerAttr(sourceDimension));
        }

        strides.push_back(rewriter.getI64IntegerAttr(1));
      }
    }

    auto resultType = mlir::memref::SubViewOp::inferRankReducedResultType(
        mlir::cast<ArrayType>(op.getResult().getType()).getShape(),
        mlir::cast<mlir::MemRefType>(adaptor.getSource().getType()), offsets,
        sizes, strides);

    mlir::Value result = rewriter.create<mlir::memref::SubViewOp>(
        loc, mlir::cast<mlir::MemRefType>(resultType), adaptor.getSource(),
        offsets, sizes, strides);

    rewriter.replaceOpWithNewOp<mlir::memref::CastOp>(
        op, getTypeConverter()->convertType(op.getResult().getType()), result);

    return mlir::success();
  }
};

class LoadOpLowering : public ModelicaOpConversionPattern<LoadOp> {
  using ModelicaOpConversionPattern<LoadOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getArray(),
                                                      adaptor.getIndices());

    return mlir::success();
  }
};

class StoreOpLowering : public ModelicaOpConversionPattern<StoreOp> {
  using ModelicaOpConversionPattern<StoreOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
        op, adaptor.getValue(), adaptor.getArray(), adaptor.getIndices());

    return mlir::success();
  }
};

class ArrayFillOpLowering : public ModelicaOpRewritePattern<ArrayFillOp> {
  using ModelicaOpRewritePattern<ArrayFillOp>::ModelicaOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayFillOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    ArrayType arrayType = op.getArrayType();

    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    mlir::Value one =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    llvm::SmallVector<mlir::Value> lowerBounds(arrayType.getRank(), zero);
    llvm::SmallVector<mlir::Value> upperBounds;
    llvm::SmallVector<mlir::Value> steps(arrayType.getRank(), one);

    for (int64_t i = 0, e = arrayType.getRank(); i < e; ++i) {
      mlir::Value dimension = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(i));

      upperBounds.push_back(
          rewriter.create<DimOp>(loc, op.getArray(), dimension));
    }

    mlir::Value value = op.getValue();

    if (mlir::Type elementType = arrayType.getElementType();
        value.getType() != elementType) {
      value = rewriter.create<CastOp>(loc, elementType, value);
    }

    mlir::scf::buildLoopNest(rewriter, loc, lowerBounds, upperBounds, steps,
                             [&](mlir::OpBuilder &nestedBuilder, mlir::Location,
                                 mlir::ValueRange position) {
                               rewriter.create<StoreOp>(
                                   loc, value, op.getArray(), position);
                             });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ArrayCopyOpDifferentTypeLowering
    : public ModelicaOpRewritePattern<ArrayCopyOp> {
  using ModelicaOpRewritePattern<ArrayCopyOp>::ModelicaOpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (!(op.getSource().getType().getElementType() !=
          op.getDestination().getType().getElementType())) {
      return rewriter.notifyMatchFailure(op, "Expected different types");
    }

    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    mlir::Value one =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    int64_t rank = op.getSource().getType().getRank();

    llvm::SmallVector<mlir::Value> lowerBounds(rank, zero);
    llvm::SmallVector<mlir::Value> upperBounds;
    llvm::SmallVector<mlir::Value> steps(rank, one);

    for (int64_t i = 0; i < rank; ++i) {
      int64_t sourceSize = op.getSource().getType().getDimSize(i);
      int64_t destinationSize = op.getDestination().getType().getDimSize(i);

      if (sourceSize != mlir::ShapedType::kDynamic) {
        upperBounds.push_back(rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(sourceSize)));
      } else if (destinationSize != mlir::ShapedType::kDynamic) {
        upperBounds.push_back(rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(destinationSize)));
      } else {
        mlir::Value dimension = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(i));

        upperBounds.push_back(
            rewriter.create<DimOp>(loc, op.getSource(), dimension));
      }
    }

    mlir::scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location,
            mlir::ValueRange position) {
          mlir::Value value =
              rewriter.create<LoadOp>(loc, op.getSource(), position);

          mlir::Type destinationElementType =
              op.getDestination().getType().getElementType();

          if (value.getType() != destinationElementType) {
            value = rewriter.create<CastOp>(loc, destinationElementType, value);
          }

          rewriter.create<StoreOp>(loc, value, op.getDestination(), position);
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ArrayCopyOpSameTypeLowering
    : public ModelicaOpConversionPattern<ArrayCopyOp> {
  using ModelicaOpConversionPattern<ArrayCopyOp>::ModelicaOpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ArrayCopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!(op.getSource().getType().getElementType() ==
          op.getDestination().getType().getElementType())) {
      return rewriter.notifyMatchFailure(op, "Expected equal types");
    }

    rewriter.replaceOpWithNewOp<mlir::memref::CopyOp>(op, adaptor.getSource(),
                                                      adaptor.getDestination());

    return mlir::success();
  }
};
} // namespace

void BaseModelicaToMemRefConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToMemRefConversionPass::convertOperations() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<ArrayCastOp>();

  target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
    return !mlir::isa<ArrayType>(op.getResult().getType());
  });

  target.addIllegalOp<ArrayToTensorOp, TensorToArrayOp>();

  target.addIllegalOp<GlobalVariableOp, GlobalVariableGetOp, AllocaOp, AllocOp,
                      ArrayFromElementsOp, ArrayBroadcastOp, FreeOp, DimOp,
                      SubscriptionOp, LoadOp, StoreOp, ArrayFillOp,
                      ArrayCopyOp>();

  target.addDynamicallyLegalOp<PoolVariableGetOp>(
      [](PoolVariableGetOp op) { return !mlir::isa<ArrayType>(op.getType()); });

  target.addIllegalOp<FillOp, NDimsOp, SizeOp>();

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::DataLayout dataLayout(moduleOp);
  TypeConverter typeConverter(&getContext(), dataLayout);
  typeConverter.addConversion([](mlir::IndexType type) { return type; });

  mlir::RewritePatternSet patterns(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;

  populateBaseModelicaToMemRefConversionPatterns(
      patterns, &getContext(), typeConverter, symbolTableCollection);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateBaseModelicaToMemRefConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  patterns.insert<ArrayCastOpLowering>(typeConverter, context);

  patterns.insert<ConstantOpArrayLowering>(typeConverter, context,
                                           symbolTableCollection);

  patterns.insert<ArrayToTensorOpLowering, TensorToArrayOpLowering>(
      typeConverter, context);

  patterns.insert<GlobalVariableOpLowering>(typeConverter, context,
                                            symbolTableCollection);

  patterns.insert<GlobalVariableGetOpLowering, PoolVariableGetOpLowering>(
      typeConverter, context);

  patterns.insert<AllocaOpLowering, AllocOpLowering>(typeConverter, context);

  patterns.insert<ArrayFromElementsOpLowering, ArrayBroadcastOpLowering>(
      context);

  patterns.insert<FreeOpLowering, DimOpLowering, SubscriptionOpLowering,
                  LoadOpLowering, StoreOpLowering>(typeConverter, context);

  patterns.insert<ArrayFillOpLowering, ArrayCopyOpDifferentTypeLowering>(
      context);

  patterns.insert<ArrayCopyOpSameTypeLowering>(typeConverter, context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToMemRefConversionPass() {
  return std::make_unique<BaseModelicaToMemRefConversionPass>();
}
} // namespace mlir
