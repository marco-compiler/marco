#include "marco/Codegen/Conversion/BaseModelicaToLinalg/BaseModelicaToLinalg.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOLINALGCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
class BaseModelicaToLinalgConversionPass
    : public mlir::impl::BaseModelicaToLinalgConversionPassBase<
          BaseModelicaToLinalgConversionPass> {
public:
  using BaseModelicaToLinalgConversionPassBase<
      BaseModelicaToLinalgConversionPass>::
      BaseModelicaToLinalgConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

void BaseModelicaToLinalgConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

static void collectDynamicDimensionSizes(
    mlir::OpBuilder &builder, mlir::Value source, mlir::TensorType tensorType,
    llvm::SmallVectorImpl<mlir::Value> &dimensionSizes) {
  if (!tensorType.hasRank()) {
    return;
  }

  for (int64_t dim = 0, rank = tensorType.getRank(); dim < rank; ++dim) {
    if (tensorType.getDimSize(dim) == mlir::ShapedType::kDynamic) {
      mlir::Value index = builder.create<mlir::arith::ConstantOp>(
          source.getLoc(), builder.getIndexAttr(dim));

      dimensionSizes.push_back(
          builder.create<mlir::tensor::DimOp>(source.getLoc(), source, index));
    }
  }
}

namespace {
class CastOpLowering : public mlir::OpConversionPattern<CastOp> {
public:
  using mlir::OpConversionPattern<CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getValue();

    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(operand.getType());

    if (!operandTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (requestedResultTensorType.getRank() != operandTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, operand, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 1> inputs;
    inputs.push_back(operand);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicSizes);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        requestedResultTensorType.getRank(), rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 10> indexingMaps(2, identityMap);

    llvm::SmallVector<mlir::utils::IteratorType, 10> iteratorTypes(
        requestedResultTensorType.getRank(),
        mlir::utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, requestedResultTensorType, inputs, destination, indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange blockArgs) {
          mlir::Value result = builder.create<CastOp>(
              loc, requestedResultTensorType.getElementType(), blockArgs[0]);

          builder.create<mlir::linalg::YieldOp>(loc, result);
        });

    return mlir::success();
  }
};

class NotOpLowering : public mlir::OpConversionPattern<NotOp> {
public:
  using mlir::OpConversionPattern<NotOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(operand.getType());

    if (!operandTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (requestedResultTensorType.getRank() != operandTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, operand, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 1> inputs;
    inputs.push_back(operand);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicSizes);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        requestedResultTensorType.getRank(), rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 10> indexingMaps(2, identityMap);

    llvm::SmallVector<mlir::utils::IteratorType, 10> iteratorTypes(
        requestedResultTensorType.getRank(),
        mlir::utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, requestedResultTensorType, inputs, destination, indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange blockArgs) {
          mlir::Value result = builder.create<NotOp>(
              loc, requestedResultTensorType.getElementType(), blockArgs[0]);

          builder.create<mlir::linalg::YieldOp>(loc, result);
        });

    return mlir::success();
  }
};

class AndOpLowering : public mlir::OpConversionPattern<AndOp> {
public:
  using mlir::OpConversionPattern<AndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicSizes);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        requestedResultTensorType.getRank(), rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 10> indexingMaps(3, identityMap);

    llvm::SmallVector<mlir::utils::IteratorType, 10> iteratorTypes(
        requestedResultTensorType.getRank(),
        mlir::utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, requestedResultTensorType, inputs, destination, indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange blockArgs) {
          mlir::Value result = builder.create<AndOp>(
              loc, requestedResultTensorType.getElementType(), blockArgs[0],
              blockArgs[1]);

          builder.create<mlir::linalg::YieldOp>(loc, result);
        });

    return mlir::success();
  }
};

class OrOpLowering : public mlir::OpConversionPattern<OrOp> {
public:
  using mlir::OpConversionPattern<OrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicSizes);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        requestedResultTensorType.getRank(), rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 10> indexingMaps(3, identityMap);

    llvm::SmallVector<mlir::utils::IteratorType, 10> iteratorTypes(
        requestedResultTensorType.getRank(),
        mlir::utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, requestedResultTensorType, inputs, destination, indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange blockArgs) {
          mlir::Value result = builder.create<OrOp>(
              loc, requestedResultTensorType.getElementType(), blockArgs[0],
              blockArgs[1]);

          builder.create<mlir::linalg::YieldOp>(loc, result);
        });

    return mlir::success();
  }
};

class NegateOpLowering : public mlir::OpConversionPattern<NegateOp> {
public:
  using mlir::OpConversionPattern<NegateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getOperand();

    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(operand.getType());

    if (!operandTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (requestedResultTensorType.getRank() != operandTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, operand, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 1> inputs;
    inputs.push_back(operand);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType, dynamicSizes);

    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        requestedResultTensorType.getRank(), rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 10> indexingMaps(2, identityMap);

    llvm::SmallVector<mlir::utils::IteratorType, 10> iteratorTypes(
        requestedResultTensorType.getRank(),
        mlir::utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        op, requestedResultTensorType, inputs, destination, indexingMaps,
        iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc,
            mlir::ValueRange blockArgs) {
          mlir::Value result = builder.create<NegateOp>(
              loc, requestedResultTensorType.getElementType(), blockArgs[0]);

          builder.create<mlir::linalg::YieldOp>(loc, result);
        });

    return mlir::success();
  }
};

class AddOpLowering : public mlir::OpConversionPattern<AddOp> {
public:
  using mlir::OpConversionPattern<AddOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!requestedResultTensorType ||
        requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          op.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          op.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType.clone(genericElementType), dynamicSizes);

    auto addOp = rewriter.create<mlir::linalg::AddOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = addOp.getResult(0);

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class AddEWOpTensorsLowering : public mlir::OpRewritePattern<AddEWOp> {
public:
  using mlir::OpRewritePattern<AddEWOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddEWOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(), lhs, rhs);

    return mlir::success();
  }
};

class AddEWOMixedLowering : public mlir::OpConversionPattern<AddEWOp> {
public:
  using mlir::OpConversionPattern<AddEWOp>::OpConversionPattern;

  mlir::LogicalResult lower(AddEWOp op,
                            mlir::ConversionPatternRewriter &rewriter,
                            mlir::Value scalar, mlir::Value tensor) const {
    mlir::Location loc = op.getLoc();
    auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(tensorType.getElementType(), scalar.getType());

    if (tensorType.getElementType() != genericElementType) {
      tensor = rewriter.create<CastOp>(
          tensor.getLoc(), tensorType.clone(genericElementType), tensor);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, tensor,
                                 mlir::cast<mlir::TensorType>(tensor.getType()),
                                 dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        scalar.getLoc(), scalar, tensor.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(tensor);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensor.getType(), dynamicSizes);

    auto addOp = rewriter.create<mlir::linalg::AddOp>(loc, tensor.getType(),
                                                      inputs, destination);

    mlir::Value result = addOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class AddEWOpScalarTensorLowering : public AddEWOMixedLowering {
public:
  using AddEWOMixedLowering::AddEWOMixedLowering;

  mlir::LogicalResult
  matchAndRewrite(AddEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(!mlir::isa<mlir::TensorType>(lhs.getType()) &&
          mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    return lower(op, rewriter, lhs, rhs);
  }
};

class AddEWOpTensorScalarLowering : public AddEWOMixedLowering {
public:
  using AddEWOMixedLowering::AddEWOMixedLowering;

  mlir::LogicalResult
  matchAndRewrite(AddEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(mlir::isa<mlir::TensorType>(lhs.getType()) &&
          !mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    return lower(op, rewriter, rhs, lhs);
  }
};

class SubOpLowering : public mlir::OpConversionPattern<SubOp> {
public:
  using mlir::OpConversionPattern<SubOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!requestedResultTensorType ||
        requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          op.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          op.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType.clone(genericElementType), dynamicSizes);

    auto subOp = rewriter.create<mlir::linalg::SubOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = subOp.getResult(0);

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class SubEWOpTensorsLowering : public mlir::OpRewritePattern<SubEWOp> {
public:
  using mlir::OpRewritePattern<SubEWOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubEWOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    rewriter.replaceOpWithNewOp<SubOp>(op, op.getResult().getType(), lhs, rhs);

    return mlir::success();
  }
};

class SubEWOpScalarTensorLowering : public mlir::OpConversionPattern<SubEWOp> {
public:
  using mlir::OpConversionPattern<SubEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(!mlir::isa<mlir::TensorType>(lhs.getType()) &&
          mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhs.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(lhs.getType(), rhsTensorType.getElementType());

    if (lhs.getType() != genericElementType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericElementType, lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          rhs.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, rhs, rhsTensorType, dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        lhs.getLoc(), lhs, rhs.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(splatOp);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, rhsTensorType, dynamicSizes);

    auto subOp = rewriter.create<mlir::linalg::SubOp>(loc, rhsTensorType,
                                                      inputs, destination);

    mlir::Value result = subOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class SubEWOpTensorScalarLowering : public mlir::OpConversionPattern<SubEWOp> {
public:
  using mlir::OpConversionPattern<SubEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(mlir::isa<mlir::TensorType>(lhs.getType()) &&
          !mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhs.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(lhsTensorType.getElementType(), rhs.getType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          lhs.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhs.getType() != genericElementType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericElementType, rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, lhsTensorType, dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        rhs.getLoc(), rhs, lhs.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, lhsTensorType, dynamicSizes);

    auto subOp = rewriter.create<mlir::linalg::SubOp>(loc, lhsTensorType,
                                                      inputs, destination);

    mlir::Value result = subOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class MulOpScalarProductLowering : public mlir::OpConversionPattern<MulOp> {
public:
  using mlir::OpConversionPattern<MulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isScalarProduct()) {
      return rewriter.notifyMatchFailure(op, "Not a scalar product");
    }

    mlir::Location loc = op.getLoc();

    mlir::Value scalar = adaptor.getLhs();
    mlir::Value tensor = adaptor.getRhs();

    auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(tensorType.getElementType(), scalar.getType());

    if (tensorType.getElementType() != genericElementType) {
      tensor = rewriter.create<CastOp>(
          tensor.getLoc(), tensorType.clone(genericElementType), tensor);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, tensor,
                                 mlir::cast<mlir::TensorType>(tensor.getType()),
                                 dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        scalar.getLoc(), scalar, tensor.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(tensor);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensor.getType(), dynamicSizes);

    auto mulOp = rewriter.create<mlir::linalg::MulOp>(loc, tensor.getType(),
                                                      inputs, destination);

    mlir::Value result = mulOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

/// Cross product of two 1-D arrays.
/// Result is a scalar.
///
/// [ x1, x2, x3 ] * [ y1, y2, y3 ] = x1 * y1 + x2 * y2 + x3 * y3
class MulOpCrossProductLowering : public mlir::OpConversionPattern<MulOp> {
public:
  using mlir::OpConversionPattern<MulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isCrossProduct()) {
      return rewriter.notifyMatchFailure(op, "Not a cross product");
    }

    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhs.getType());

    auto requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          lhs.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          rhs.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    mlir::Type resultType = mlir::RankedTensorType::get({}, genericElementType);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultType, mlir::ValueRange());

    auto dotOp = rewriter.create<mlir::linalg::DotOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = dotOp.getResult(0);
    result = rewriter.create<mlir::tensor::ExtractOp>(result.getLoc(), result);

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

// clang-format off
/// Product of a vector (1-D array) and a matrix (2-D array).
///
/// [ x1 ] * [ y11, y12 ] = [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
/// [ x2 ]   [ y21, y22 ]
/// [ x3 ]   [ y31, y32 ]
// clang-format on
class MulOpVectorMatrixLowering : public mlir::OpConversionPattern<MulOp> {
public:
  using mlir::OpConversionPattern<MulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isVectorMatrixProduct()) {
      return rewriter.notifyMatchFailure(op, "Not a vector-matrix product");
    }

    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhs.getType());

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          lhs.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          rhs.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, rhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType.clone(genericElementType), dynamicSizes);

    auto vecMatOp =
        rewriter.create<mlir::linalg::VecmatOp>(loc, inputs, destination);

    mlir::Value result = vecMatOp.getResult(0);

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

/// Product of a matrix (2-D array) and a vector (1-D array).
///
/// [ x11, x12 ] * [ y1, y2 ] = [ x11 * y1 + x12 * y2 ]
/// [ x21, x22 ]                [ x21 * y1 + x22 * y2 ]
/// [ x31, x32 ]                [ x31 * y1 + x22 * y2 ]
class MulOpMatrixVectorLowering : public mlir::OpRewritePattern<MulOp> {
public:
  using mlir::OpRewritePattern<MulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, mlir::PatternRewriter &rewriter) const override {
    if (!op.isMatrixVectorProduct()) {
      return rewriter.notifyMatchFailure(op, "Not a matrix-vector product");
    }

    mlir::Location loc = op.getLoc();

    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    lhs = rewriter.create<TransposeOp>(loc, lhs);

    rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(), rhs, lhs);

    return mlir::success();
  }
};

// clang-format off
/// Product of two matrices (2-D arrays).
///
/// [ x11, x12, x13 ] * [ y11, y12 ] = [ x11 * y11 + x12 * y21 + x13 * y31, x11 * y12 + x12 * y22 + x13 * y32 ]
/// [ x21, x22, x23 ]   [ y21, y22 ]   [ x21 * y11 + x22 * y21 + x23 * y31, x21 * y12 + x22 * y22 + x23 * y32 ]
/// [ x31, x32, x33 ]   [ y31, y32 ]   [ x31 * y11 + x32 * y21 + x33 * y31, x31 * y12 + x32 * y22 + x33 * y32 ]
/// [ x41, x42, x43 ]
// clang-format on
class MulOpMatrixLowering : public mlir::OpConversionPattern<MulOp> {
public:
  using mlir::OpConversionPattern<MulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.isMatrixProduct()) {
      return rewriter.notifyMatchFailure(op, "Not a matrix product");
    }

    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhs.getType());

    auto requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          lhs.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          rhs.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    auto resultTensorType =
        mlir::cast<mlir::TensorType>(
            getTypeConverter()->convertType(op.getResult().getType()))
            .clone(genericElementType);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    llvm::SmallVector<mlir::Value, 2> dynamicSizes;

    if (resultTensorType.getDimSize(0) == mlir::ShapedType::kDynamic) {
      mlir::Value dimIndex = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(0));

      dynamicSizes.push_back(
          rewriter.create<mlir::tensor::DimOp>(loc, lhs, dimIndex));
    }

    if (resultTensorType.getDimSize(1) == mlir::ShapedType::kDynamic) {
      mlir::Value dimIndex = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(1));

      dynamicSizes.push_back(
          rewriter.create<mlir::tensor::DimOp>(loc, rhs, dimIndex));
    }

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultTensorType, dynamicSizes);

    auto matmulOp = rewriter.create<mlir::linalg::MatmulOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = matmulOp.getResult(0);

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class MulEWOpTensorsLowering : public mlir::OpConversionPattern<MulEWOp> {
public:
  using mlir::OpConversionPattern<MulEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!requestedResultTensorType ||
        requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          op.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          op.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType.clone(genericElementType), dynamicSizes);

    auto mulOp = rewriter.create<mlir::linalg::MulOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = mulOp.getResult(0);

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class MulEWOMixedLowering : public mlir::OpConversionPattern<MulEWOp> {
public:
  using mlir::OpConversionPattern<MulEWOp>::OpConversionPattern;

  mlir::LogicalResult lower(MulEWOp op,
                            mlir::ConversionPatternRewriter &rewriter,
                            mlir::Value scalar, mlir::Value tensor) const {
    mlir::Location loc = op.getLoc();
    auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(tensorType.getElementType(), scalar.getType());

    if (tensorType.getElementType() != genericElementType) {
      tensor = rewriter.create<CastOp>(
          tensor.getLoc(), tensorType.clone(genericElementType), tensor);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, tensor,
                                 mlir::cast<mlir::TensorType>(tensor.getType()),
                                 dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        scalar.getLoc(), scalar, tensor.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(tensor);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensor.getType(), dynamicSizes);

    auto mulOp = rewriter.create<mlir::linalg::MulOp>(loc, tensor.getType(),
                                                      inputs, destination);

    mlir::Value result = mulOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class MulEWOpScalarTensorLowering : public MulEWOMixedLowering {
public:
  using MulEWOMixedLowering::MulEWOMixedLowering;

  mlir::LogicalResult
  matchAndRewrite(MulEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(!mlir::isa<mlir::TensorType>(lhs.getType()) &&
          mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    return lower(op, rewriter, lhs, rhs);
  }
};

class MulEWOpTensorScalarLowering : public MulEWOMixedLowering {
public:
  using MulEWOMixedLowering::MulEWOMixedLowering;

  mlir::LogicalResult
  matchAndRewrite(MulEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(mlir::isa<mlir::TensorType>(lhs.getType()) &&
          !mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    return lower(op, rewriter, rhs, lhs);
  }
};

class DivOpTensorScalarLowering : public mlir::OpConversionPattern<DivOp> {
public:
  using mlir::OpConversionPattern<DivOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(mlir::isa<mlir::TensorType>(lhs.getType()) &&
          !mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value tensor = lhs;
    mlir::Value scalar = rhs;

    auto tensorType = mlir::cast<mlir::TensorType>(tensor.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(tensorType.getElementType(), scalar.getType());

    if (tensorType.getElementType() != genericElementType) {
      tensor = rewriter.create<CastOp>(
          tensor.getLoc(), tensorType.clone(genericElementType), tensor);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, tensor,
                                 mlir::cast<mlir::TensorType>(tensor.getType()),
                                 dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        scalar.getLoc(), scalar, tensor.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(tensor);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensor.getType(), dynamicSizes);

    auto subOp = rewriter.create<mlir::linalg::DivOp>(loc, tensor.getType(),
                                                      inputs, destination);

    mlir::Value result = subOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class DivEWOpTensorsLowering : public mlir::OpConversionPattern<DivEWOp> {
public:
  using mlir::OpConversionPattern<DivEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    auto lhsTensorType = mlir::dyn_cast<mlir::TensorType>(lhs.getType());
    auto rhsTensorType = mlir::dyn_cast<mlir::TensorType>(rhs.getType());

    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    if (lhsTensorType.getRank() != rhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible ranks");
    }

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (!requestedResultTensorType ||
        requestedResultTensorType.getRank() != lhsTensorType.getRank()) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Type genericElementType = getMostGenericScalarType(
        lhsTensorType.getElementType(), rhsTensorType.getElementType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          op.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          op.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, requestedResultTensorType,
                                 dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultTensorType.clone(genericElementType), dynamicSizes);

    auto divOp = rewriter.create<mlir::linalg::DivOp>(
        op.getLoc(), destination.getType(), inputs, destination);

    mlir::Value result = divOp.getResult(0);

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class DivEWOpScalarTensorLowering : public mlir::OpConversionPattern<DivEWOp> {
public:
  using mlir::OpConversionPattern<DivEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(!mlir::isa<mlir::TensorType>(lhs.getType()) &&
          mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    auto rhsTensorType = mlir::cast<mlir::TensorType>(rhs.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(lhs.getType(), rhsTensorType.getElementType());

    if (lhs.getType() != genericElementType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericElementType, lhs);
    }

    if (rhsTensorType.getElementType() != genericElementType) {
      rhs = rewriter.create<CastOp>(
          rhs.getLoc(), rhsTensorType.clone(genericElementType), rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, rhs, rhsTensorType, dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        lhs.getLoc(), lhs, rhs.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(splatOp);
    inputs.push_back(rhs);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, rhsTensorType, dynamicSizes);

    auto divOp = rewriter.create<mlir::linalg::DivOp>(loc, rhsTensorType,
                                                      inputs, destination);

    mlir::Value result = divOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class DivEWOpTensorScalarLowering : public mlir::OpConversionPattern<DivEWOp> {
public:
  using mlir::OpConversionPattern<DivEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!(mlir::isa<mlir::TensorType>(lhs.getType()) &&
          !mlir::isa<mlir::TensorType>(rhs.getType()))) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    auto lhsTensorType = mlir::cast<mlir::TensorType>(lhs.getType());

    mlir::Type genericElementType =
        getMostGenericScalarType(lhsTensorType.getElementType(), rhs.getType());

    if (lhsTensorType.getElementType() != genericElementType) {
      lhs = rewriter.create<CastOp>(
          lhs.getLoc(), lhsTensorType.clone(genericElementType), lhs);
    }

    if (rhs.getType() != genericElementType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericElementType, rhs);
    }

    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    collectDynamicDimensionSizes(rewriter, lhs, lhsTensorType, dynamicSizes);

    auto splatOp = rewriter.create<mlir::tensor::SplatOp>(
        rhs.getLoc(), rhs, lhs.getType(), dynamicSizes);

    llvm::SmallVector<mlir::Value, 2> inputs;
    inputs.push_back(lhs);
    inputs.push_back(splatOp);

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, lhsTensorType, dynamicSizes);

    auto divOp = rewriter.create<mlir::linalg::DivOp>(loc, lhsTensorType,
                                                      inputs, destination);

    mlir::Value result = divOp.getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class PowOpLowering : public mlir::OpConversionPattern<PowOp> {
public:
  using mlir::OpConversionPattern<PowOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(PowOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value base = adaptor.getBase();
    mlir::Value exponent = adaptor.getExponent();

    auto baseTensorType = mlir::dyn_cast<mlir::TensorType>(base.getType());

    if (!baseTensorType) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Type baseElementType = baseTensorType.getElementType();
    llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

    for (int64_t dim = 0, rank = baseTensorType.getRank(); dim < rank; ++dim) {
      if (baseTensorType.getDimSize(dim) == mlir::ShapedType::kDynamic) {
        dynamicDimensions.push_back(
            rewriter.create<mlir::tensor::DimOp>(loc, base, dim));
      }
    }

    mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(1));

    if (one.getType() != baseElementType) {
      one = rewriter.create<CastOp>(loc, baseElementType, one);
    }

    mlir::Value onesMatrix = rewriter.create<mlir::tensor::SplatOp>(
        loc, one, baseTensorType, dynamicDimensions);

    if (!mlir::isa<mlir::IndexType>(exponent.getType())) {
      exponent =
          rewriter.create<CastOp>(loc, rewriter.getIndexType(), exponent);
    }

    mlir::Value lowerBound =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    mlir::Value step =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    auto forOp = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, exponent,
                                                   step, onesMatrix);

    rewriter.setInsertionPointToStart(forOp.getBody());

    llvm::SmallVector<mlir::Value, 2> matmulArgs;
    matmulArgs.push_back(forOp.getRegionIterArg(0));
    matmulArgs.push_back(base);

    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(0));

    if (zero.getType() != baseElementType) {
      zero = rewriter.create<CastOp>(loc, baseElementType, zero);
    }

    mlir::Value destination = rewriter.create<mlir::tensor::SplatOp>(
        loc, zero, baseTensorType, dynamicDimensions);

    auto matmulOp = rewriter.create<mlir::linalg::MatmulOp>(
        loc, forOp.getRegionIterArg(0).getType(), matmulArgs, destination);

    rewriter.create<mlir::scf::YieldOp>(loc, matmulOp.getResultTensors());
    mlir::Value result = forOp.getResult(0);

    rewriter.setInsertionPointAfter(forOp);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class TransposeOpLowering : public mlir::OpConversionPattern<TransposeOp> {
public:
  using mlir::OpConversionPattern<TransposeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value matrix = adaptor.getMatrix();

    auto resultTensorType =
        mlir::cast<mlir::TensorType>(op.getResult().getType());

    int64_t rank = resultTensorType.getRank();
    llvm::SmallVector<mlir::Value, 10> dynamicSizes;

    for (int64_t dim = 0; dim < rank; ++dim) {
      if (resultTensorType.getDimSize(dim) == mlir::ShapedType::kDynamic) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(rank - dim - 1));

        dynamicSizes.push_back(
            rewriter.create<mlir::tensor::DimOp>(op.getLoc(), matrix, index));
      }
    }

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resultTensorType, dynamicSizes);

    llvm::SmallVector<int64_t, 2> permutation;
    permutation.push_back(1);
    permutation.push_back(0);

    auto transposeOp = rewriter.create<mlir::linalg::TransposeOp>(
        op.getLoc(), matrix, destination, permutation);

    mlir::Value result = transposeOp->getResult(0);

    auto requestedResultTensorType = mlir::cast<mlir::TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    if (result.getType() != requestedResultTensorType) {
      result = rewriter.create<CastOp>(result.getLoc(),
                                       requestedResultTensorType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
} // namespace

namespace {
template <typename Op>
void addDynamicallyLegalVectorizedOneOperandAndResultOp(
    mlir::ConversionTarget &target) {
  target.addDynamicallyLegalOp<Op>([](Op op) {
    if (op->getResults().size() != 1) {
      return true;
    }

    return !mlir::isa<mlir::TensorType>(op->getResult(0).getType());
  });
}

template <typename Op1, typename Op2, typename... Opn>
void addDynamicallyLegalVectorizedOneOperandAndResultOp(
    mlir::ConversionTarget &target) {
  addDynamicallyLegalVectorizedOneOperandAndResultOp<Op1>(target);
  addDynamicallyLegalVectorizedOneOperandAndResultOp<Op2, Opn...>(target);
}
} // namespace

mlir::LogicalResult BaseModelicaToLinalgConversionPass::convertOperations() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();

  target.addDynamicallyLegalOp<CastOp>([](CastOp op) {
    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getValue().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !operandTensorType || !resultTensorType ||
           resultTensorType.getRank() != operandTensorType.getRank();
  });

  target.addDynamicallyLegalOp<NotOp>([](NotOp op) {
    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getOperand().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !operandTensorType || !resultTensorType ||
           resultTensorType.getRank() != operandTensorType.getRank();
  });

  target.addDynamicallyLegalOp<AndOp>([](AndOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !lhsTensorType || !rhsTensorType ||
           lhsTensorType.getRank() != rhsTensorType.getRank() ||
           !resultTensorType ||
           resultTensorType.getRank() != lhsTensorType.getRank();
  });

  target.addDynamicallyLegalOp<OrOp>([](OrOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !lhsTensorType || !rhsTensorType ||
           lhsTensorType.getRank() != rhsTensorType.getRank() ||
           !resultTensorType ||
           resultTensorType.getRank() != lhsTensorType.getRank();
  });

  target.addDynamicallyLegalOp<NegateOp>([](NegateOp op) {
    auto operandTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getOperand().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !operandTensorType || !resultTensorType ||
           resultTensorType.getRank() != operandTensorType.getRank();
  });

  target.addDynamicallyLegalOp<AddOp>([](AddOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !lhsTensorType || !rhsTensorType || !resultTensorType ||
           lhsTensorType.getRank() != rhsTensorType.getRank() ||
           resultTensorType.getRank() != lhsTensorType.getRank();
  });

  target.addDynamicallyLegalOp<AddEWOp>([](AddEWOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    if (lhsTensorType && rhsTensorType && resultTensorType &&
        lhsTensorType.getRank() == rhsTensorType.getRank() &&
        resultTensorType.getRank() == lhsTensorType.getRank()) {
      return false;
    }

    if (lhsTensorType && !rhsTensorType && resultTensorType &&
        lhsTensorType.getRank() == resultTensorType.getRank()) {
      return false;
    }

    if (!lhsTensorType && rhsTensorType && resultTensorType &&
        rhsTensorType.getRank() == resultTensorType.getRank()) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<SubOp>([](SubOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    return !lhsTensorType || !rhsTensorType || !resultTensorType ||
           lhsTensorType.getRank() != rhsTensorType.getRank() ||
           resultTensorType.getRank() != lhsTensorType.getRank();
  });

  target.addDynamicallyLegalOp<SubEWOp>([](SubEWOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    auto resultTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getResult().getType());

    if (lhsTensorType && rhsTensorType && resultTensorType &&
        lhsTensorType.getRank() == rhsTensorType.getRank() &&
        resultTensorType.getRank() == lhsTensorType.getRank()) {
      return false;
    }

    if (lhsTensorType && !rhsTensorType && resultTensorType &&
        lhsTensorType.getRank() == resultTensorType.getRank()) {
      return false;
    }

    if (!lhsTensorType && rhsTensorType && resultTensorType &&
        rhsTensorType.getRank() == resultTensorType.getRank()) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<MulOp>([](MulOp op) {
    if (op.isScalarProduct() || op.isCrossProduct() ||
        op.isVectorMatrixProduct() || op.isMatrixVectorProduct() ||
        op.isMatrixProduct()) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<MulEWOp>([](MulEWOp op) {
    auto lhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getLhs().getType());
    auto rhsTensorType =
        mlir::dyn_cast<mlir::TensorType>(op.getRhs().getType());

    if (lhsTensorType || rhsTensorType) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<DivOp>([](DivOp op) {
    if (mlir::isa<mlir::TensorType>(op.getLhs().getType())) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<DivEWOp>([](DivEWOp op) {
    if (mlir::isa<mlir::TensorType>(op.getLhs().getType()) ||
        mlir::isa<mlir::TensorType>(op.getRhs().getType())) {
      return false;
    }

    return true;
  });

  target.addDynamicallyLegalOp<PowOp>([](PowOp op) {
    if (mlir::isa<mlir::TensorType>(op.getBase().getType())) {
      return false;
    }

    return true;
  });

  target.addIllegalOp<TransposeOp>();

  addDynamicallyLegalVectorizedOneOperandAndResultOp<
      AbsOp, AcosOp, AsinOp, AtanOp, CeilOp, CosOp, CoshOp, ExpOp, FloorOp,
      IntegerOp, LogOp, Log10Op, SignOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp>(
      target);

  target.addDynamicallyLegalOp<Atan2Op>([](Atan2Op op) {
    return !mlir::isa<mlir::TensorType>(op.getResult().getType());
  });

  target.markUnknownOpDynamicallyLegal(
      [](mlir::Operation *op) { return true; });

  mlir::DataLayout dataLayout(moduleOp);
  TypeConverter typeConverter(&getContext(), dataLayout);

  mlir::RewritePatternSet patterns(&getContext());

  populateBaseModelicaToLinalgConversionPatterns(patterns, &getContext(),
                                                 typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace {
template <typename Op>
struct VectorizedSingleOperandAndResultOpLowering
    : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;
  using OpAdaptor = typename mlir::OpConversionPattern<Op>::OpAdaptor;

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (op->getResults().size() != 1) {
      return rewriter.notifyMatchFailure(op, "Multiple results found");
    }

    if (!mlir::isa<mlir::TensorType>(op->getResult(0).getType())) {
      return rewriter.notifyMatchFailure(op, "Not a vectorized op");
    }

    mlir::Value operand = adaptor.getOperands()[0];

    mlir::Type requestedResultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());

    auto requestedResultTensorType =
        mlir::cast<mlir::TensorType>(requestedResultType);

    llvm::SmallVector<mlir::Value> dynamicDimensions;

    for (int64_t dim = 0, rank = requestedResultTensorType.getRank();
         dim < rank; ++dim) {
      if (requestedResultTensorType.getDimSize(dim) ==
          mlir::ShapedType::kDynamic) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(dim));

        mlir::Value dimensionSize =
            rewriter.create<mlir::tensor::DimOp>(loc, operand, index);

        dynamicDimensions.push_back(dimensionSize);
      }
    }

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultType, dynamicDimensions);

    llvm::SmallVector<mlir::Value> scalarResults;

    rewriter.replaceOpWithNewOp<mlir::linalg::MapOp>(
        op, operand, destination,
        [&](mlir::OpBuilder &builder, mlir::Location nestedLoc,
            mlir::ValueRange scalarArgs) {
          auto scalarOp = builder.create<Op>(
              nestedLoc, requestedResultTensorType.getElementType(),
              scalarArgs[0]);

          builder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                scalarOp->getResults());
        });

    return mlir::success();
  }
};

struct VectorizedAtan2OpLowering : public mlir::OpConversionPattern<Atan2Op> {
  using mlir::OpConversionPattern<Atan2Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Atan2Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (!mlir::isa<mlir::TensorType>(op.getResult().getType())) {
      return rewriter.notifyMatchFailure(op, "Not a vectorized op");
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    auto requestedResultTensorType =
        mlir::cast<mlir::TensorType>(requestedResultType);

    llvm::SmallVector<mlir::Value> dynamicDimensions;

    for (int64_t dim = 0, rank = requestedResultTensorType.getRank();
         dim < rank; ++dim) {
      if (requestedResultTensorType.getDimSize(dim) ==
          mlir::ShapedType::kDynamic) {
        mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(dim));

        mlir::Value dimensionSize =
            rewriter.create<mlir::tensor::DimOp>(loc, adaptor.getY(), index);

        dynamicDimensions.push_back(dimensionSize);
      }
    }

    mlir::Value destination = rewriter.create<mlir::tensor::EmptyOp>(
        loc, requestedResultType, dynamicDimensions);

    llvm::SmallVector<mlir::Value> scalarResults;

    rewriter.replaceOpWithNewOp<mlir::linalg::MapOp>(
        op, adaptor.getOperands(), destination,
        [&](mlir::OpBuilder &builder, mlir::Location nestedLoc,
            mlir::ValueRange scalarArgs) {
          auto scalarOp = builder.create<Atan2Op>(
              nestedLoc, requestedResultTensorType.getElementType(),
              scalarArgs[0], scalarArgs[1]);

          builder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                scalarOp->getResults());
        });

    return mlir::success();
  }
};
} // namespace

namespace {
template <typename... Op>
void insertVectorizedOneOperandAndResultPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns.insert<VectorizedSingleOperandAndResultOpLowering<Op>...>(
      typeConverter, context);
}
} // namespace

namespace mlir {
void populateBaseModelicaToLinalgConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns.insert<CastOpLowering>(typeConverter, context);

  patterns.insert<NotOpLowering, AndOpLowering, OrOpLowering, NegateOpLowering,
                  AddOpLowering>(typeConverter, context);

  patterns.insert<AddEWOpTensorsLowering>(context);

  patterns.insert<AddEWOpScalarTensorLowering, AddEWOpTensorScalarLowering,
                  SubOpLowering>(typeConverter, context);

  patterns.insert<SubEWOpTensorsLowering>(context);

  patterns.insert<SubEWOpScalarTensorLowering, SubEWOpTensorScalarLowering>(
      typeConverter, context);

  patterns.insert<MulOpScalarProductLowering, MulOpCrossProductLowering,
                  MulOpVectorMatrixLowering, MulOpMatrixLowering>(typeConverter,
                                                                  context);

  patterns.insert<MulOpMatrixVectorLowering>(context);

  patterns.insert<MulEWOpTensorsLowering, MulEWOpScalarTensorLowering,
                  MulEWOpTensorScalarLowering>(typeConverter, context);

  patterns.insert<DivOpTensorScalarLowering, DivEWOpTensorsLowering,
                  DivEWOpScalarTensorLowering, DivEWOpTensorScalarLowering>(
      typeConverter, context);

  patterns.insert<PowOpLowering>(typeConverter, context);

  patterns.insert<TransposeOpLowering>(typeConverter, context);

  // Patterns for vectorized operations.
  insertVectorizedOneOperandAndResultPatterns<
      AbsOp, AcosOp, AsinOp, AtanOp, CeilOp, CosOp, CoshOp, ExpOp, FloorOp,
      IntegerOp, LogOp, Log10Op, SignOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp>(
      patterns, context, typeConverter);

  patterns.insert<VectorizedAtan2OpLowering>(typeConverter, context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToLinalgConversionPass() {
  return std::make_unique<BaseModelicaToLinalgConversionPass>();
}
} // namespace mlir
