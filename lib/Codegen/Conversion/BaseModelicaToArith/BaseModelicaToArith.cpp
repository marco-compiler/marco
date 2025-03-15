#include "marco/Codegen/Conversion/BaseModelicaToArith/BaseModelicaToArith.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/TypeConverter.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_BASEMODELICATOARITHCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
} // namespace mlir

using namespace ::mlir::bmodelica;

namespace {
class BaseModelicaToArithConversionPass
    : public mlir::impl::BaseModelicaToArithConversionPassBase<
          BaseModelicaToArithConversionPass> {
public:
  using BaseModelicaToArithConversionPassBase<
      BaseModelicaToArithConversionPass>::BaseModelicaToArithConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertOperations();
};
} // namespace

//===---------------------------------------------------------------------===//
// Cast operations
//===---------------------------------------------------------------------===//

namespace {
struct CastOpIndexLowering : public mlir::OpConversionPattern<CastOp> {
  using mlir::OpConversionPattern<CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getValue();
    mlir::Type operandType = operand.getType();

    if (!mlir::isa<mlir::IndexType>(operandType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            resultType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    if (mlir::isa<mlir::IndexType>(resultType)) {
      rewriter.replaceOp(op, operand);
      return mlir::success();
    }

    mlir::Value result = operand;
    auto requestedBitWidth = resultType.getIntOrFloatBitWidth();

    result = rewriter.create<mlir::arith::IndexCastOp>(
        loc, rewriter.getIntegerType(requestedBitWidth), result);

    if (mlir::isa<mlir::FloatType>(resultType)) {
      if (requestedBitWidth == 1) {
        result =
            rewriter.create<mlir::arith::UIToFPOp>(loc, resultType, result);
      } else {
        result =
            rewriter.create<mlir::arith::SIToFPOp>(loc, resultType, result);
      }
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct CastOpIntegerLowering : public mlir::OpConversionPattern<CastOp> {
  using mlir::OpConversionPattern<CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getValue();
    mlir::Type operandType = operand.getType();

    if (!mlir::isa<mlir::IntegerType>(operandType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType>(resultType)) {
      rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, resultType,
                                                            operand);

      return mlir::success();
    }

    if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(resultType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Value result = operand;
    auto bitWidth = result.getType().getIntOrFloatBitWidth();
    auto requestedBitWidth = resultType.getIntOrFloatBitWidth();

    mlir::Type requestedBitWidthIntegerType =
        rewriter.getIntegerType(requestedBitWidth);

    if (bitWidth < requestedBitWidth) {
      if (bitWidth == 1) {
        result = rewriter.create<mlir::arith::ExtUIOp>(
            loc, requestedBitWidthIntegerType, result);
      } else {
        result = rewriter.create<mlir::arith::ExtSIOp>(
            loc, requestedBitWidthIntegerType, result);
      }
    } else if (bitWidth > requestedBitWidth) {
      result = rewriter.create<mlir::arith::TruncIOp>(
          loc, requestedBitWidthIntegerType, result);
    }

    if (mlir::isa<mlir::FloatType>(resultType)) {
      if (requestedBitWidth == 1) {
        result =
            rewriter.create<mlir::arith::UIToFPOp>(loc, resultType, result);
      } else {
        result =
            rewriter.create<mlir::arith::SIToFPOp>(loc, resultType, result);
      }
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct CastOpFloatLowering : public mlir::OpConversionPattern<CastOp> {
  using mlir::OpConversionPattern<CastOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value operand = adaptor.getValue();
    mlir::Type operandType = operand.getType();

    if (!mlir::isa<mlir::FloatType>(operandType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    mlir::Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            resultType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible result type");
    }

    mlir::Value result = operand;
    auto bitWidth = result.getType().getIntOrFloatBitWidth();
    auto requestedBitWidth = resultType.getIntOrFloatBitWidth();

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(resultType)) {
      if (bitWidth == 1) {
        result = rewriter.create<mlir::arith::FPToUIOp>(
            loc, rewriter.getIntegerType(1), result);
      } else {
        result = rewriter.create<mlir::arith::FPToSIOp>(
            loc, rewriter.getIntegerType(requestedBitWidth), result);
      }
    }

    if (mlir::isa<mlir::IndexType>(resultType)) {
      result =
          rewriter.create<mlir::arith::IndexCastOp>(loc, resultType, result);
    }

    if (mlir::isa<mlir::FloatType>(resultType)) {
      if (bitWidth < requestedBitWidth) {
        result = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, result);
      } else if (bitWidth > requestedBitWidth) {
        result =
            rewriter.create<mlir::arith::TruncFOp>(loc, resultType, result);
      }
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Range operations
//===---------------------------------------------------------------------===//

namespace {
struct RangeSizeOpLowering : public mlir::OpRewritePattern<RangeSizeOp> {
  using mlir::OpRewritePattern<RangeSizeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(RangeSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value beginValue = rewriter.create<RangeBeginOp>(loc, op.getRange());

    mlir::Value endValue = rewriter.create<RangeEndOp>(loc, op.getRange());

    mlir::Value stepValue = rewriter.create<RangeStepOp>(loc, op.getRange());

    mlir::Value result = rewriter.create<SubOp>(
        loc, op.getRange().getType().getInductionType(), endValue, beginValue);

    result =
        rewriter.create<DivOp>(loc, rewriter.getIndexType(), result, stepValue);

    mlir::Value one =
        rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    result = rewriter.create<AddOp>(loc, rewriter.getIndexType(), result, one);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

namespace {
struct EqOpLowering : public mlir::OpConversionPattern<EqOp> {
  using mlir::OpConversionPattern<EqOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(EqOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::eq, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OEQ, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct NotEqOpLowering : public mlir::OpConversionPattern<NotEqOp> {
  using mlir::OpConversionPattern<NotEqOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NotEqOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct GtOpLowering : public mlir::OpConversionPattern<GtOp> {
  using mlir::OpConversionPattern<GtOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible left-hand side operands");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operands");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sgt, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGT, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct GteOpLowering : public mlir::OpConversionPattern<GteOp> {
  using mlir::OpConversionPattern<GteOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(GteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible left-hand side operands");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operands");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sge, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGE, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct LtOpLowering : public mlir::OpConversionPattern<LtOp> {
  using mlir::OpConversionPattern<LtOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LtOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible left-hand side operands");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operands");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::slt, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLT, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct LteOpLowering : public mlir::OpConversionPattern<LteOp> {
  using mlir::OpConversionPattern<LteOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(LteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible left-hand side operands");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operands");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(lhs.getLoc(), genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(rhs.getLoc(), genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sle, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLE, lhs,
          rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(loc, requestedResultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

namespace {
struct NotOpIntegerLowering : public mlir::OpConversionPattern<NotOp> {
  using mlir::OpConversionPattern<NotOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value operand = adaptor.getOperand();
    mlir::Type operandType = operand.getType();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(operandType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }

    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

    mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::eq,
        operand, zero);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result = rewriter.create<CastOp>(loc, requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct NotOpFloatLowering : public mlir::OpConversionPattern<NotOp> {
  using mlir::OpConversionPattern<NotOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value operand = adaptor.getOperand();
    mlir::Type operandType = operand.getType();

    if (!mlir::isa<mlir::FloatType>(operandType)) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

    mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OEQ,
        adaptor.getOperand(), zero);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AndOpIntegersLikeLowering : public mlir::OpConversionPattern<AndOp> {
  using mlir::OpConversionPattern<AndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(lhs.getType()) ||
        !mlir::isa<mlir::IndexType, mlir::IntegerType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AndOpFloatsLowering : public mlir::OpConversionPattern<AndOp> {
  using mlir::OpConversionPattern<AndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::FloatType>(lhs.getType()) ||
        !mlir::isa<mlir::FloatType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AndOpIntegerLikeFloatLowering : public mlir::OpConversionPattern<AndOp> {
  using mlir::OpConversionPattern<AndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::FloatType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct AndOpFloatIntegerLikeLowering : public mlir::OpConversionPattern<AndOp> {
  using mlir::OpConversionPattern<AndOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::FloatType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct OrOpIntegersLikeLowering : public mlir::OpConversionPattern<OrOp> {
  using mlir::OpConversionPattern<OrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct OrOpFloatsLowering : public mlir::OpConversionPattern<OrOp> {
  using mlir::OpConversionPattern<OrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::FloatType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::FloatType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct OrOpIntegerLikeFloatLowering : public mlir::OpConversionPattern<OrOp> {
  using mlir::OpConversionPattern<OrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::FloatType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct OrOpFloatIntegerLikeLowering : public mlir::OpConversionPattern<OrOp> {
  using mlir::OpConversionPattern<OrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::FloatType>(lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType>(rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

    mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, lhs,
        lhsZero);

    mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

    mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
        loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, rhs,
        rhsZero);

    mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
        loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

namespace {
struct ConstantOpLowering : public mlir::OpConversionPattern<ConstantOp> {
public:
  using mlir::OpConversionPattern<ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::TypedAttr attribute = convertAttribute(
        rewriter, getTypeConverter()->convertType(op.getResult().getType()),
        op.getValue());

    if (!attribute) {
      return rewriter.notifyMatchFailure(op, "Incompatible attribute");
    }

    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attribute);
    return mlir::success();
  }

private:
  mlir::TypedAttr convertAttribute(mlir::OpBuilder &builder,
                                   mlir::Type resultType,
                                   mlir::TypedAttr attribute) const {
    if (mlir::isa<mlir::TensorType>(resultType)) {
      return convertTensorTypedAttribute(builder, resultType, attribute);
    }

    if (mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            attribute.getType())) {
      return attribute;
    }

    if (auto booleanAttribute = mlir::dyn_cast<BooleanAttr>(attribute)) {
      return builder.getBoolAttr(booleanAttribute.getValue());
    }

    if (auto integerAttribute = mlir::dyn_cast<IntegerAttr>(attribute)) {
      return builder.getIntegerAttr(resultType,
                                    integerAttribute.getValue().getSExtValue());
    }

    if (auto realAttribute = mlir::dyn_cast<RealAttr>(attribute)) {
      return builder.getFloatAttr(resultType, realAttribute.getValue());
    }

    return {};
  }

  mlir::TypedAttr convertTensorTypedAttribute(mlir::OpBuilder &builder,
                                              mlir::Type resultType,
                                              mlir::TypedAttr attribute) const {
    if (mlir::isa<mlir::DenseIntOrFPElementsAttr>(attribute)) {
      return attribute;
    }

    if (auto denseBooleanAttr =
            mlir::dyn_cast<DenseBooleanElementsAttr>(attribute)) {
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(resultType),
          denseBooleanAttr.getValues());
    }

    if (auto denseIntegerAttr =
            mlir::dyn_cast<DenseIntegerElementsAttr>(attribute)) {
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(resultType),
          denseIntegerAttr.getValues());
    }

    if (auto denseRealAttr = mlir::dyn_cast<DenseRealElementsAttr>(attribute)) {
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(resultType), denseRealAttr.getValues());
    }

    return {};
  }
};

struct NegateOpLowering : public mlir::OpConversionPattern<NegateOp> {
  using mlir::OpConversionPattern<NegateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Type operandType = adaptor.getOperand().getType();

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(operandType)) {
      mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(operandType));

      mlir::Value result = rewriter.create<mlir::arith::SubIOp>(
          loc, zeroValue, adaptor.getOperand());

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(operandType)) {
      mlir::Value result =
          rewriter.create<mlir::arith::NegFOp>(loc, adaptor.getOperand());

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Incompatible operand");
  }
};

struct AddOpLowering : public mlir::OpConversionPattern<AddOp> {
  using mlir::OpConversionPattern<AddOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(loc, genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(loc, genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::AddIOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::AddFOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct AddEWOpLowering : public mlir::OpConversionPattern<AddEWOp> {
  using mlir::OpConversionPattern<AddEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(AddEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(),
                                       op.getLhs(), op.getRhs());

    return mlir::success();
  }
};

struct SubOpLowering : public mlir::OpConversionPattern<SubOp> {
  using mlir::OpConversionPattern<SubOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(loc, genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(loc, genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::SubIOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::SubFOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct SubEWOpLowering : public mlir::OpConversionPattern<SubEWOp> {
  using mlir::OpConversionPattern<SubEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SubEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(),
                                       op.getLhs(), op.getRhs());

    return mlir::success();
  }
};

struct MulOpLowering : public mlir::OpConversionPattern<MulOp> {
  using mlir::OpConversionPattern<MulOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(loc, genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(loc, genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::MulIOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::MulFOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct MulEWOpLowering : public mlir::OpConversionPattern<MulEWOp> {
  using mlir::OpConversionPattern<MulEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(MulEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(),
                                       op.getLhs(), op.getRhs());

    return mlir::success();
  }
};

struct DivOpLowering : public mlir::OpConversionPattern<DivOp> {
  using mlir::OpConversionPattern<DivOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    mlir::Type genericType =
        getMostGenericScalarType(lhs.getType(), rhs.getType());

    if (lhs.getType() != genericType) {
      lhs = rewriter.create<CastOp>(loc, genericType, lhs);
    }

    if (rhs.getType() != genericType) {
      rhs = rewriter.create<CastOp>(loc, genericType, rhs);
    }

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResult().getType());

    if (mlir::isa<mlir::IndexType, mlir::IntegerType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::DivSIOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (mlir::isa<mlir::FloatType>(genericType)) {
      mlir::Value result = rewriter.create<mlir::arith::DivFOp>(loc, lhs, rhs);

      if (result.getType() != requestedResultType) {
        result = rewriter.create<CastOp>(result.getLoc(), requestedResultType,
                                         result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct DivEWOpLowering : public mlir::OpConversionPattern<DivEWOp> {
  using mlir::OpConversionPattern<DivEWOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(DivEWOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value lhs = adaptor.getLhs();
    mlir::Value rhs = adaptor.getRhs();

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            lhs.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "Incompatible left-hand side operand");
    }

    if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
            rhs.getType())) {
      return rewriter.notifyMatchFailure(
          op, "Incompatible right-hand side operand");
    }

    rewriter.replaceOpWithNewOp<DivOp>(op, op.getResult().getType(),
                                       op.getLhs(), op.getRhs());

    return mlir::success();
  }
};

struct ReductionOpLowering : public mlir::OpConversionPattern<ReductionOp> {
  using mlir::OpConversionPattern<ReductionOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ReductionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (llvm::any_of(op.getIterables(), [](mlir::Value iterable) {
          return !mlir::isa<RangeType>(iterable.getType());
        })) {
      return mlir::failure();
    }

    // Determine the initial value of the accumulator.
    auto yieldOp = mlir::cast<YieldOp>(op.getBody()->getTerminator());

    mlir::Type resultType =
        getTypeConverter()->convertType(yieldOp.getValues()[0].getType());

    llvm::SmallVector<mlir::Value, 1> initialValues;

    if (mlir::failed(getInitialValues(rewriter, loc, op.getAction(), resultType,
                                      initialValues))) {
      return mlir::failure();
    }

    // Compute the boundaries and steps so that the step becomes positive.
    llvm::SmallVector<mlir::Value, 6> lowerBounds;
    llvm::SmallVector<mlir::Value, 6> upperBounds;
    llvm::SmallVector<mlir::Value, 6> steps;

    mlir::Value zero =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));

    mlir::Value one =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

    mlir::Type indexType = rewriter.getIndexType();

    for (mlir::Value iterable : op.getIterables()) {
      mlir::Value step = rewriter.create<RangeStepOp>(loc, iterable);

      mlir::Value isNonNegativeStep = rewriter.create<GteOp>(
          loc, BooleanType::get(rewriter.getContext()), step, zero);

      mlir::Value beginValue = rewriter.create<RangeBeginOp>(loc, iterable);

      mlir::Value size = rewriter.create<RangeSizeOp>(loc, iterable);

      // begin + (size - 1) * step
      mlir::Value actualEndValue = rewriter.create<AddOp>(
          loc, indexType, beginValue,
          rewriter.create<MulOp>(
              loc, indexType, step,
              rewriter.create<SubOp>(loc, indexType, size, one)));

      auto lowerBoundSelectOp = rewriter.create<SelectOp>(
          loc, indexType, isNonNegativeStep, beginValue, actualEndValue);

      auto upperBoundSelectOp = rewriter.create<SelectOp>(
          loc, indexType, isNonNegativeStep, actualEndValue, beginValue);

      mlir::Value negatedStep = rewriter.create<NegateOp>(loc, indexType, step);

      auto stepSelectOp = rewriter.create<SelectOp>(
          loc, indexType, isNonNegativeStep, step, negatedStep);

      lowerBounds.push_back(lowerBoundSelectOp.getResult(0));

      mlir::Value upperBound = upperBoundSelectOp.getResult(0);

      upperBound =
          rewriter.create<AddOp>(loc, upperBound.getType(), upperBound, one);

      upperBounds.push_back(upperBound);
      steps.push_back(stepSelectOp.getResult(0));
    }

    // Create the parallel loops.
    auto parallelOp = rewriter.create<mlir::scf::ParallelOp>(
        loc, lowerBounds, upperBounds, steps, initialValues);

    // Create the reduce operation.
    rewriter.setInsertionPoint(yieldOp);

    mlir::Value currentElementValue = yieldOp.getValues()[0];

    currentElementValue = getTypeConverter()->materializeTargetConversion(
        rewriter, loc, resultType, currentElementValue);

    auto reduceOp =
        rewriter.create<mlir::scf::ReduceOp>(loc, currentElementValue);

    rewriter.setInsertionPointToEnd(&reduceOp.getReductions().front().front());

    mlir::Value reductionResult =
        computeReductionResult(rewriter, loc, op.getAction(), resultType,
                               reduceOp.getReductions().front().getArgument(0),
                               reduceOp.getReductions().front().getArgument(1));

    rewriter.create<mlir::scf::ReduceReturnOp>(loc, reductionResult);
    rewriter.eraseOp(yieldOp);

    // Inline the old body.
    llvm::SmallVector<mlir::Value, 6> newInductions;
    rewriter.setInsertionPointToStart(parallelOp.getBody());

    for (auto [oldInduction, newInduction] :
         llvm::zip(op.getInductions(), parallelOp.getInductionVars())) {
      mlir::Value mapped = newInduction;

      if (mapped.getType() != oldInduction.getType()) {
        mapped = rewriter.create<CastOp>(
            loc, getTypeConverter()->convertType(oldInduction.getType()),
            mapped);

        mapped = getTypeConverter()->materializeSourceConversion(
            rewriter, loc, oldInduction.getType(), mapped);
      }

      newInductions.push_back(mapped);
    }

    rewriter.mergeBlocks(op.getBody(), parallelOp.getBody(), newInductions);

    rewriter.replaceOp(op, parallelOp);
    return mlir::success();
  }

private:
  mlir::LogicalResult
  getInitialValues(mlir::OpBuilder &builder, mlir::Location loc,
                   llvm::StringRef action, mlir::Type resultType,
                   llvm::SmallVectorImpl<mlir::Value> &initialValues) const {
    if (action == "add") {
      if (mlir::isa<mlir::IntegerType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getIntegerAttr(resultType, 0)));

        return mlir::success();
      }

      if (mlir::isa<mlir::FloatType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getFloatAttr(resultType, 0)));

        return mlir::success();
      }

      if (mlir::isa<mlir::IndexType>(resultType)) {
        initialValues.push_back(
            builder.create<ConstantOp>(loc, builder.getIndexAttr(0)));

        return mlir::success();
      }
    }

    if (action == "mul") {
      if (mlir::isa<mlir::IntegerType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getIntegerAttr(resultType, 1)));

        return mlir::success();
      }

      if (mlir::isa<mlir::FloatType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getFloatAttr(resultType, 1)));

        return mlir::success();
      }

      if (mlir::isa<mlir::IndexType>(resultType)) {
        initialValues.push_back(
            builder.create<ConstantOp>(loc, builder.getIndexAttr(1)));

        return mlir::success();
      }
    }

    if (action == "min") {
      if (mlir::isa<mlir::IntegerType>(resultType)) {
        unsigned int bitWidth = resultType.getIntOrFloatBitWidth();

        if (bitWidth == 8) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int8_t>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 16) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int16_t>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 32) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int32_t>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 64) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int64_t>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }
      }

      if (mlir::isa<mlir::FloatType>(resultType)) {
        unsigned int bitWidth = resultType.getIntOrFloatBitWidth();

        if (bitWidth == 32) {
          mlir::Attribute value = builder.getFloatAttr(
              resultType, std::numeric_limits<float>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 64) {
          mlir::Attribute value = builder.getFloatAttr(
              resultType, std::numeric_limits<double>::max());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }
      }

      if (mlir::isa<mlir::IndexType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getIndexAttr(std::numeric_limits<int64_t>::max())));

        return mlir::success();
      }
    }

    if (action == "max") {
      if (mlir::isa<mlir::IntegerType>(resultType)) {
        unsigned int bitWidth = resultType.getIntOrFloatBitWidth();

        if (bitWidth == 8) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int8_t>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 16) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int16_t>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 32) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int32_t>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 64) {
          mlir::Attribute value = builder.getIntegerAttr(
              resultType, std::numeric_limits<int64_t>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }
      }

      if (mlir::isa<mlir::FloatType>(resultType)) {
        unsigned int bitWidth = resultType.getIntOrFloatBitWidth();

        if (bitWidth == 32) {
          mlir::Attribute value = builder.getFloatAttr(
              resultType, std::numeric_limits<float>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }

        if (bitWidth == 64) {
          mlir::Attribute value = builder.getFloatAttr(
              resultType, std::numeric_limits<double>::min());

          initialValues.push_back(builder.create<ConstantOp>(loc, value));
          return mlir::success();
        }
      }

      if (mlir::isa<mlir::IndexType>(resultType)) {
        initialValues.push_back(builder.create<ConstantOp>(
            loc, builder.getIndexAttr(std::numeric_limits<int64_t>::min())));

        return mlir::success();
      }
    }

    return mlir::failure();
  }

  mlir::Value computeReductionResult(mlir::OpBuilder &builder,
                                     mlir::Location loc, llvm::StringRef action,
                                     mlir::Type resultType, mlir::Value lhs,
                                     mlir::Value rhs) const {
    if (action == "add") {
      return builder.create<AddOp>(loc, resultType, lhs, rhs);
    }

    if (action == "mul") {
      return builder.create<MulOp>(loc, resultType, lhs, rhs);
    }

    if (action == "min") {
      return builder.create<MinOp>(loc, resultType, lhs, rhs);
    }

    if (action == "max") {
      return builder.create<MaxOp>(loc, resultType, lhs, rhs);
    }

    llvm_unreachable("Unexpected reduction kind");
    return nullptr;
  }
};
} // namespace

//===---------------------------------------------------------------------===//
// Various operations
//===---------------------------------------------------------------------===//

namespace {
struct SelectOpLowering : public mlir::OpConversionPattern<SelectOp> {
  using mlir::OpConversionPattern<SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (adaptor.getTrueValues().size() > 1 ||
        adaptor.getFalseValues().size() > 1) {
      return rewriter.notifyMatchFailure(
          op, "Multiple values among true or false cases");
    }

    assert(adaptor.getTrueValues().size() == adaptor.getFalseValues().size());

    mlir::Value trueValue = adaptor.getTrueValues()[0];
    mlir::Value falseValue = adaptor.getFalseValues()[0];

    auto trueValueTensorType =
        mlir::dyn_cast<mlir::TensorType>(trueValue.getType());

    auto falseValueTensorType =
        mlir::dyn_cast<mlir::TensorType>(falseValue.getType());

    if (trueValueTensorType && falseValueTensorType) {
      mlir::Type genericElementType =
          getMostGenericScalarType(trueValueTensorType.getElementType(),
                                   falseValueTensorType.getElementType());

      if (trueValueTensorType.getElementType() != genericElementType) {
        trueValue = rewriter.create<CastOp>(
            trueValue.getLoc(), trueValueTensorType.clone(genericElementType),
            trueValue);
      }

      if (falseValueTensorType.getElementType() != genericElementType) {
        falseValue = rewriter.create<CastOp>(
            falseValue.getLoc(), falseValueTensorType.clone(genericElementType),
            falseValue);
      }
    } else if (!trueValueTensorType && !falseValueTensorType) {
      mlir::Type genericType =
          getMostGenericScalarType(trueValue.getType(), falseValue.getType());

      if (falseValue.getType() != genericType) {
        trueValue =
            rewriter.create<CastOp>(trueValue.getLoc(), genericType, trueValue);
      }

      if (falseValue.getType() != genericType) {
        falseValue = rewriter.create<CastOp>(falseValue.getLoc(), genericType,
                                             falseValue);
      }
    } else {
      return rewriter.notifyMatchFailure(op, "Incompatible operands");
    }

    mlir::Value condition = adaptor.getCondition();

    if (condition.getType() != rewriter.getI1Type()) {
      condition = rewriter.create<CastOp>(condition.getLoc(),
                                          rewriter.getI1Type(), condition);
    }

    mlir::Value result = rewriter.create<mlir::arith::SelectOp>(
        loc, condition, trueValue, falseValue);

    mlir::Type requestedResultType =
        getTypeConverter()->convertType(op.getResults()[0].getType());

    if (result.getType() != requestedResultType) {
      result =
          rewriter.create<CastOp>(result.getLoc(), requestedResultType, result);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct SelectOpMultipleValuesPattern : public mlir::OpRewritePattern<SelectOp> {
  using mlir::OpRewritePattern<SelectOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SelectOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();

    if (!(op.getTrueValues().size() > 1 || op.getFalseValues().size() > 1)) {
      return rewriter.notifyMatchFailure(op, "Expected multiple operands");
    }

    mlir::TypeRange resultTypes = op.getResultTypes();
    llvm::SmallVector<mlir::Value> results;

    assert(resultTypes.size() == op.getTrueValues().size());
    assert(resultTypes.size() == op.getFalseValues().size());

    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      mlir::Value trueValue = op.getTrueValues()[i];
      mlir::Value falseValue = op.getFalseValues()[i];

      auto resultOp = rewriter.create<SelectOp>(
          loc, resultTypes[i], op.getCondition(), trueValue, falseValue);

      results.push_back(resultOp.getResult(0));
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};
} // namespace

void BaseModelicaToArithConversionPass::runOnOperation() {
  if (mlir::failed(convertOperations())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult BaseModelicaToArithConversionPass::convertOperations() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<BaseModelicaDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::scf::SCFDialect>();

  target.addDynamicallyLegalOp<CastOp>([](CastOp op) {
    if (mlir::isa<BooleanType, IntegerType, RealType, mlir::IndexType,
                  mlir::IntegerType, mlir::FloatType>(
            op.getValue().getType())) {
      return false;
    }

    return true;
  });

  target.addIllegalOp<RangeSizeOp>();

  target.addIllegalOp<EqOp, NotEqOp, GtOp, GteOp, LtOp, LteOp>();

  target.addIllegalOp<NotOp, AndOp, OrOp>();

  target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
    mlir::Type resultType = op.getResult().getType();
    return mlir::isa<ArrayType, RangeType>(resultType);
  });

  target.addIllegalOp<NegateOp, AddOp, AddEWOp, SubOp, SubEWOp, MulOp, MulEWOp,
                      DivOp, DivEWOp>();

  target.addDynamicallyLegalOp<PowOp>(
      [](PowOp op) { return !mlir::isa<ArrayType>(op.getBase().getType()); });

  target.addIllegalOp<ReductionOp, SelectOp>();

  mlir::DataLayout dataLayout(moduleOp);
  TypeConverter typeConverter(&getContext(), dataLayout);

  mlir::RewritePatternSet patterns(&getContext());

  populateBaseModelicaToArithConversionPatterns(patterns, &getContext(),
                                                typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

namespace mlir {
void populateBaseModelicaToArithConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context,
    mlir::TypeConverter &typeConverter) {
  patterns
      .insert<CastOpIndexLowering, CastOpIntegerLowering, CastOpFloatLowering>(
          typeConverter, context);

  patterns.insert<RangeSizeOpLowering>(context);

  patterns.insert<EqOpLowering, NotEqOpLowering, GtOpLowering, GteOpLowering,
                  LtOpLowering, LteOpLowering>(typeConverter, context);

  patterns.insert<NotOpIntegerLowering, NotOpFloatLowering,
                  AndOpIntegersLikeLowering, AndOpFloatsLowering,
                  AndOpIntegerLikeFloatLowering, AndOpFloatIntegerLikeLowering,
                  OrOpIntegersLikeLowering, OrOpFloatsLowering,
                  OrOpIntegerLikeFloatLowering, OrOpFloatIntegerLikeLowering>(
      typeConverter, context);

  patterns.insert<AddEWOpLowering, SubEWOpLowering, MulEWOpLowering,
                  DivEWOpLowering>(context);

  // TODO use runtime library
  // patterns.insert<
  //    PowOpMatrixLowering>(context, assertions);

  patterns.insert<ConstantOpLowering, NegateOpLowering, AddOpLowering,
                  SubOpLowering, MulOpLowering, DivOpLowering>(typeConverter,
                                                               context);

  patterns.insert<ReductionOpLowering, SelectOpLowering>(typeConverter,
                                                         context);

  patterns.insert<SelectOpMultipleValuesPattern>(context);
}

std::unique_ptr<mlir::Pass> createBaseModelicaToArithConversionPass() {
  return std::make_unique<BaseModelicaToArithConversionPass>();
}
} // namespace mlir
