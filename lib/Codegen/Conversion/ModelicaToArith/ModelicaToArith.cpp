#include "marco/Codegen/Conversion/ModelicaToArith/ModelicaToArith.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include <limits>

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOARITHCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static void iterateArray(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value array,
    llvm::function_ref<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)> callback)
{
  assert(array.getType().isa<ArrayType>());
  auto arrayType = array.getType().cast<ArrayType>();

  mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::Value, 3> lowerBounds(arrayType.getRank(), zero);

  llvm::SmallVector<mlir::Value, 3> upperBounds;

  for (unsigned int i = 0, e = arrayType.getRank(); i < e; ++i) {
    mlir::Value dim = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(i));
    upperBounds.push_back(builder.create<DimOp>(loc, array, dim));
  }

  mlir::Value one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
  llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

  // Create nested loops in order to iterate on each dimension of the array
  mlir::scf::buildLoopNest(builder, loc, lowerBounds, upperBounds, steps, callback);
}

static bool isScalarType(mlir::Type type)
{
  return type.isa<
      mlir::IndexType, mlir::IntegerType, mlir::FloatType,
      BooleanType, IntegerType, RealType>();
}

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

      std::pair<mlir::Value, mlir::Value> castToMostGenericType(
          mlir::OpBuilder& builder, mlir::Value lhs, mlir::Value rhs) const
      {
        mlir::Type resultType =
            getMostGenericType(lhs.getType(), rhs.getType());

        lhs = castToType(builder, lhs, resultType);
        rhs = castToType(builder, rhs, resultType);

        return std::make_pair(lhs, rhs);
      }

      mlir::Type getMostGenericType(mlir::Type lhs, mlir::Type rhs) const
      {
        if (lhs.isa<mlir::FloatType>() && rhs.isa<mlir::FloatType>()) {
          if (lhs.getIntOrFloatBitWidth() > rhs.getIntOrFloatBitWidth()) {
            return lhs;
          } else {
            return rhs;
          }
        } else if (lhs.isa<mlir::FloatType>()) {
          return lhs;
        } else if (rhs.isa<mlir::FloatType>()) {
          return rhs;
        } else if (lhs.isa<mlir::IndexType>()) {
          return lhs;
        } else if (rhs.isa<mlir::IndexType>()) {
          return rhs;
        }

        assert(lhs.isa<mlir::IntegerType>() && rhs.isa<mlir::IntegerType>());

        if (lhs.getIntOrFloatBitWidth() > rhs.getIntOrFloatBitWidth()) {
          return lhs;
        }

        return rhs;
      }

      mlir::Value castToType(
          mlir::OpBuilder& builder,
          mlir::Value value,
          mlir::Type resultType) const
      {
        if (resultType.isa<mlir::FloatType>()) {
          value = castToFloat(builder, value, resultType);
        } else if (resultType.isa<mlir::IndexType>()) {
          value = castToIndex(builder, value, resultType);
        } else if (resultType.isa<mlir::IntegerType>()) {
          value = castToInteger(builder, value, resultType);
        }

        return value;
      }

      mlir::Value castToFloat(
          mlir::OpBuilder& builder,
          mlir::Value value,
          mlir::Type resultType) const
      {
        mlir::Location loc = value.getLoc();
        assert(resultType.isa<mlir::FloatType>());
        unsigned int resultBitWidth = resultType.getIntOrFloatBitWidth();

        if (mlir::Type type = value.getType(); type.isa<mlir::IndexType>()) {
          value = builder.create<mlir::arith::IndexCastOp>(
              loc, builder.getIntegerType(resultBitWidth), value);
        }

        if (mlir::Type type = value.getType(); type.isa<mlir::IntegerType>()) {
          unsigned int sourceBitWidth = type.getIntOrFloatBitWidth();

          if (sourceBitWidth < resultBitWidth) {
            if (sourceBitWidth == 1) {
              value = builder.create<mlir::arith::ExtUIOp>(
                  loc, builder.getIntegerType(resultBitWidth), value);
            } else {
              value = builder.create<mlir::arith::ExtSIOp>(
                  loc, builder.getIntegerType(resultBitWidth), value);
            }
          } else if (sourceBitWidth > resultBitWidth) {
            value = builder.create<mlir::arith::TruncIOp>(
                loc, builder.getIntegerType(resultBitWidth), value);
          }

          value = builder.create<mlir::arith::SIToFPOp>(
              loc, resultType, value);
        }

        mlir::Type type = value.getType();
        assert(type.isa<mlir::FloatType>());

        if (type.getIntOrFloatBitWidth() < resultBitWidth) {
          value = builder.create<mlir::arith::ExtFOp>(loc, resultType, value);
        } else if (type.getIntOrFloatBitWidth() > resultBitWidth) {
          value = builder.create<mlir::arith::TruncFOp>(
              loc, resultType, value);
        }

        return value;
      }

      mlir::Value castToIndex(
          mlir::OpBuilder& builder,
          mlir::Value value,
          mlir::Type resultType) const
      {
        mlir::Location loc = value.getLoc();
        assert(resultType.isa<mlir::IndexType>());

        if (mlir::Type type = value.getType(); type.isa<mlir::FloatType>()) {
          unsigned int sourceBitWidth = type.getIntOrFloatBitWidth();

          value = builder.create<mlir::arith::FPToSIOp>(
              loc, builder.getIntegerType(sourceBitWidth), value);
        }

        if (mlir::Type type = value.getType(); type.isa<mlir::IntegerType>()) {
          value = builder.create<mlir::arith::IndexCastOp>(
              loc, resultType, value);
        }

        return value;
      }

      mlir::Value castToInteger(
          mlir::OpBuilder& builder,
          mlir::Value value,
          mlir::Type resultType) const
      {
        mlir::Location loc = value.getLoc();
        assert(resultType.isa<mlir::IntegerType>());
        unsigned int resultBitWidth = resultType.getIntOrFloatBitWidth();

        if (mlir::Type type = value.getType(); type.isa<mlir::FloatType>()) {
          unsigned int sourceBitWidth = type.getIntOrFloatBitWidth();

          value = builder.create<mlir::arith::FPToSIOp>(
              loc, builder.getIntegerType(sourceBitWidth), value);
        }

        if (mlir::Type type = value.getType(); type.isa<mlir::IndexType>()) {
          value = builder.create<mlir::arith::IndexCastOp>(
              loc, resultType, value);
        }

        mlir::Type type = value.getType();
        unsigned int sourceBitWidth = type.getIntOrFloatBitWidth();

        if (sourceBitWidth < resultBitWidth) {
          if (sourceBitWidth == 1) {
            value = builder.create<mlir::arith::ExtUIOp>(
                loc, builder.getIntegerType(resultBitWidth), value);
          } else {
            value = builder.create<mlir::arith::ExtSIOp>(
                loc, builder.getIntegerType(resultBitWidth), value);
          }
        } else if (sourceBitWidth > resultBitWidth) {
          value = builder.create<mlir::arith::TruncIOp>(
              loc, builder.getIntegerType(resultBitWidth), value);
        }

        return value;
      }
  };
}

//===---------------------------------------------------------------------===//
// Range operations
//===---------------------------------------------------------------------===//

namespace
{
  struct RangeSizeOpLowering
      : public ModelicaOpRewritePattern<RangeSizeOp>
  {
      using ModelicaOpRewritePattern<RangeSizeOp>::ModelicaOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          RangeSizeOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Value beginValue =
            rewriter.create<RangeBeginOp>(loc, op.getRange());

        mlir::Value endValue = rewriter.create<RangeEndOp>(loc, op.getRange());

        mlir::Value stepValue =
            rewriter.create<RangeStepOp>(loc, op.getRange());

        mlir::Value result = rewriter.create<SubOp>(
            loc, op.getRange().getType().getInductionType(),
            endValue, beginValue);

        result = rewriter.create<DivOp>(
            loc, rewriter.getIndexType(), result, stepValue);

        mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        result = rewriter.create<AddOp>(
            loc, rewriter.getIndexType(), result, one);

        rewriter.replaceOp(op, result);
        return mlir::success();
      }
  };
}

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

namespace
{
  struct EqOpLowering : public ModelicaOpConversionPattern<EqOp>
  {
    using ModelicaOpConversionPattern<EqOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        EqOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::eq,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::OEQ,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct NotEqOpLowering : public ModelicaOpConversionPattern<NotEqOp>
  {
    using ModelicaOpConversionPattern<NotEqOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        NotEqOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::ne,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::ONE,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct GtOpLowering : public ModelicaOpConversionPattern<GtOp>
  {
    using ModelicaOpConversionPattern<GtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        GtOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::sgt,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::OGT,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct GteOpLowering : public ModelicaOpConversionPattern<GteOp>
  {
    using ModelicaOpConversionPattern<GteOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        GteOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::sge,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::OGE,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct LtOpLowering : public ModelicaOpConversionPattern<LtOp>
  {
    using ModelicaOpConversionPattern<LtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LtOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::slt,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::OLT,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct LteOpLowering : public ModelicaOpConversionPattern<LteOp>
  {
    using ModelicaOpConversionPattern<LteOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        LteOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (mlir::Type type = lhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (mlir::Type type = rhs.getType();
          !type.isa<
              BooleanType, IntegerType, RealType,
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpIPredicate::sle,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1),
            mlir::arith::CmpFPredicate::OLE,
            lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };
}

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

namespace
{
  struct NotOpIntegerLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        NotOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value operand = adaptor.getOperand();
      mlir::Type operandType = operand.getType();

      if (!operandType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Unsupported operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::eq,
          operand, zero);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct NotOpFloatLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        NotOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value operand = adaptor.getOperand();
      mlir::Type operandType = operand.getType();

      if (!operandType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Unsupported operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::OEQ,
          adaptor.getOperand(), zero);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct NotOpArrayLowering : public ModelicaOpRewritePattern<NotOp>
  {
    using ModelicaOpRewritePattern<NotOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        NotOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      // Check if the operand is compatible.
      if (!op.getOperand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array");
      }

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();

      auto dynamicDimensions = getArrayDynamicDimensions(
          rewriter, loc, op.getOperand());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array element.
      iterateArray(
          rewriter, loc, result,
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            mlir::Value value = nestedBuilder.create<LoadOp>(
                location, op.getOperand(), indices);

            mlir::Value negated = nestedBuilder.create<NotOp>(
                location, resultArrayType.getElementType(), value);

            nestedBuilder.create<StoreOp>(location, negated, result, indices);
          });

      return mlir::success();
    }
  };

  template<typename Op>
  class BinaryLogicOpArrayLowering : public ModelicaOpRewritePattern<Op>
  {
    public:
      BinaryLogicOpArrayLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern<Op>(context), assertions(assertions)
      {
      }

      using ModelicaOpRewritePattern<Op>::ModelicaOpRewritePattern;

      virtual mlir::Value getLhs(Op op) const = 0;
      virtual mlir::Value getRhs(Op op) const = 0;

      virtual mlir::Value scalarize(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::Type elementType,
          mlir::Value lhs,
          mlir::Value rhs) const = 0;

      mlir::LogicalResult matchAndRewrite(
          Op op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        // Check if the operands are compatible.
        if (!getLhs(op).getType().template isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side operand is not an array");
        }

        if (!getRhs(op).getType().template isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side operand is not an array");
        }

        auto lhsArrayType = getLhs(op).getType().template cast<ArrayType>();
        auto rhsArrayType = getRhs(op).getType().template cast<ArrayType>();
        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        if (assertions) {
          // Check if the dimensions are compatible
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
            if (lhsShape[i] == ArrayType::kDynamic ||
                rhsShape[i] == ArrayType::kDynamic) {
              mlir::Value dimensionIndex =
                  rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIndexAttr(i));

              mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                  loc, getLhs(op), dimensionIndex);

              mlir::Value rhsDimensionSize = rewriter.create<DimOp>(
                  loc, getRhs(op), dimensionIndex);

              mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq,
                  lhsDimensionSize, rhsDimensionSize);

              rewriter.create<mlir::cf::AssertOp>(
                  loc, condition,
                  rewriter.getStringAttr("Incompatible dimensions"));
            }
          }
        }

        // Allocate the result array.
        auto resultArrayType =
            op.getResult().getType().template cast<ArrayType>();

        auto lhsDynamicDimensions = getArrayDynamicDimensions(
            rewriter, loc, getLhs(op));

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, lhsDynamicDimensions);

        // Apply the operation on each array position.
        iterateArray(
            rewriter, loc, result,
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location location,
                mlir::ValueRange indices) {
              mlir::Value lhs = nestedBuilder.create<LoadOp>(
                  loc, getLhs(op), indices);

              mlir::Value rhs = nestedBuilder.create<LoadOp>(
                  loc, getRhs(op), indices);

              mlir::Value scalarResult = scalarize(
                  nestedBuilder, location,
                  resultArrayType.getElementType(),
                  lhs, rhs);

              nestedBuilder.create<StoreOp>(
                  loc, scalarResult, result, indices);
            });

        return mlir::success();
      }

    private:
      bool assertions;
  };

  struct AndOpIntegersLikeLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AndOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct AndOpFloatsLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AndOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct AndOpIntegerLikeFloatLowering
      : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AndOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct AndOpFloatIntegerLikeLowering
      : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AndOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct AndOpArrayLowering : public BinaryLogicOpArrayLowering<AndOp>
  {
    using BinaryLogicOpArrayLowering<AndOp>::BinaryLogicOpArrayLowering;

    mlir::Value getLhs(AndOp op) const override
    {
      return op.getLhs();
    }

    mlir::Value getRhs(AndOp op) const override
    {
      return op.getRhs();
    }

    mlir::Value scalarize(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::Type elementType,
        mlir::Value lhs,
        mlir::Value rhs) const override
    {
      return builder.create<AndOp>(loc, elementType, lhs, rhs);
    }
  };

  struct OrOpIntegersLikeLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        OrOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct OrOpFloatsLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        OrOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct OrOpIntegerLikeFloatLowering
      : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        OrOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct OrOpFloatIntegerLikeLowering
      : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        OrOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhsCmp = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          lhs, lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhsCmp = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          rhs, rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhsCmp, rhsCmp);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResult().getType()));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct OrOpArrayLowering : public BinaryLogicOpArrayLowering<OrOp>
  {
    using BinaryLogicOpArrayLowering<OrOp>::BinaryLogicOpArrayLowering;

    mlir::Value getLhs(OrOp op) const override
    {
      return op.getLhs();
    }

    mlir::Value getRhs(OrOp op) const override
    {
      return op.getRhs();
    }

    mlir::Value scalarize(
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::Type elementType,
        mlir::Value lhs,
        mlir::Value rhs) const override
    {
      return builder.create<OrOp>(loc, elementType, lhs, rhs);
    }
  };
}

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

namespace
{
  struct ConstantOpScalarLowering
      : public ModelicaOpConversionPattern<ConstantOp>
  {
    public:
      using ModelicaOpConversionPattern<ConstantOp>
          ::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          ConstantOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::TypedAttr attribute = convertAttribute(
            rewriter, op.getResult().getType(),
            mlir::cast<mlir::Attribute>(op.getValue()));

        if (!attribute) {
          return rewriter.notifyMatchFailure(
              op, "Unknown attribute conversion");
        }

        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attribute);
        return mlir::success();
      }

    private:
      mlir::TypedAttr convertAttribute(
        mlir::OpBuilder& builder,
        mlir::Type resultType,
        mlir::Attribute attribute) const
      {
        if (attribute.cast<mlir::TypedAttr>().getType().isa<
            mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
          return attribute.cast<mlir::TypedAttr>();
        }

        resultType = getTypeConverter()->convertType(resultType);

        if (auto booleanAttribute = attribute.dyn_cast<BooleanAttr>()) {
          return builder.getBoolAttr(booleanAttribute.getValue());
        }

        if (auto integerAttribute = attribute.dyn_cast<IntegerAttr>()) {
          return builder.getIntegerAttr(
              resultType, integerAttribute.getValue());
        }

        if (auto realAttribute = attribute.dyn_cast<RealAttr>()) {
          return builder.getFloatAttr(resultType, realAttribute.getValue());
        }

        return {};
      }
  };

  struct NegateOpLowering : public ModelicaOpConversionPattern<NegateOp>
  {
    using ModelicaOpConversionPattern<NegateOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        NegateOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      mlir::Type type = adaptor.getOperand().getType();

      // Compute the result.
      if (type.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getZeroAttr(type));

        mlir::Value result = rewriter.create<mlir::arith::SubIOp>(
            loc, zeroValue, adaptor.getOperand());

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      if (type.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::NegFOp>(
            loc, adaptor.getOperand());

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Incompatible operand");
    }
  };

  struct NegateOpArrayLowering : public ModelicaOpRewritePattern<NegateOp>
  {
    using ModelicaOpRewritePattern<NegateOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        NegateOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      // Check if the operand is compatible.
      if (!op.getOperand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Value is not an array");
      }

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();

      auto dynamicDimensions = getArrayDynamicDimensions(
          rewriter, loc, op.getOperand());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position.
      iterateArray(
          rewriter, loc, result,
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            mlir::Value source = nestedBuilder.create<LoadOp>(
                location, op.getOperand(), indices);

            mlir::Value value = nestedBuilder.create<NegateOp>(
                location, resultArrayType.getElementType(), source);

            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddOpLowering : public ModelicaOpConversionPattern<AddOp>
  {
    using ModelicaOpConversionPattern<AddOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        AddOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<
          mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::AddIOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::AddFOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct AddOpArraysLowering : public ModelicaOpRewritePattern<AddOp>
  {
    public:
      AddOpArraysLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern(context), assertions(assertions)
      {
      }

      mlir::LogicalResult match(AddOp op) const override
      {
        mlir::Type lhsType = op.getLhs().getType();
        mlir::Type rhsType = op.getRhs().getType();

        if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
          return mlir::failure();
        }

        auto lhsArrayType = lhsType.cast<ArrayType>();
        auto rhsArrayType = rhsType.cast<ArrayType>();

        if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
          // Incompatible ranks.
          return mlir::failure();
        }

        for (const auto& [lhsDimension, rhsDimension] : llvm::zip(
                 lhsArrayType.getShape(), rhsArrayType.getShape())) {
          if (lhsDimension != ArrayType::kDynamic && rhsDimension !=
                  ArrayType::kDynamic && lhsDimension != rhsDimension) {
            // Incompatible array dimensions.
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      void rewrite(AddOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          assert(lhsArrayType.getRank() == rhsArrayType.getRank());

          for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
            if (lhsShape[i] == ArrayType::kDynamic ||
                rhsShape[i] == ArrayType::kDynamic) {
              mlir::Value dimensionIndex =
                  rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIndexAttr(i));

              mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getLhs(), dimensionIndex);

              mlir::Value rhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getRhs(), dimensionIndex);

              mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq,
                  lhsDimensionSize, rhsDimensionSize);

              rewriter.create<mlir::cf::AssertOp>(
                  loc, condition,
                  rewriter.getStringAttr("Incompatible dimensions"));
            }
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();

        auto dynamicDimensions =
            getArrayDynamicDimensions(rewriter, loc, op.getLhs());

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Apply the operation on each array position.
        iterateArray(
            rewriter, loc, result,
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location location,
                mlir::ValueRange indices) {
              mlir::Value lhs = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value rhs = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<AddOp>(
                  location, resultArrayType.getElementType(), lhs, rhs);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            });
      }

    private:
      bool assertions;
  };

  struct AddEWOpScalarsLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values.
      return mlir::LogicalResult::success(
          ::isScalarType(lhsType) && ::isScalarType(rhsType));
    }

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(
          op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct AddEWOpArraysLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be arrays.
      return mlir::LogicalResult::success(
          lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>());
    }

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(
          op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct AddEWOpMixedLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (lhsType.isa<ArrayType>() && ::isScalarType(rhsType)) {
        return mlir::success();
      }

      if (::isScalarType(lhsType) && rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      mlir::Value array = op.getLhs().getType().isa<ArrayType>()
          ? op.getLhs() : op.getRhs();

      mlir::Value scalar = op.getLhs().getType().isa<ArrayType>()
          ? op.getRhs() : op.getLhs();

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position.
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                location, array, indices);

            mlir::Value value = nestedBuilder.create<AddOp>(
                location, resultArrayType.getElementType(),
                arrayValue, scalar);

            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  struct SubOpLowering : public ModelicaOpConversionPattern<SubOp>
  {
    using ModelicaOpConversionPattern<SubOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        SubOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::SubIOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::SubFOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct SubOpArraysLowering : public ModelicaOpRewritePattern<SubOp>
  {
    public:
      SubOpArraysLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern(context), assertions(assertions)
      {
      }

      mlir::LogicalResult match(SubOp op) const override
      {
        mlir::Type lhsType = op.getLhs().getType();
        mlir::Type rhsType = op.getRhs().getType();

        if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
          return mlir::failure();
        }

        auto lhsArrayType = lhsType.cast<ArrayType>();
        auto rhsArrayType = rhsType.cast<ArrayType>();

        if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
          // Incompatible ranks.
          return mlir::failure();
        }

        for (const auto& [lhsDimension, rhsDimension] : llvm::zip(
                 lhsArrayType.getShape(), rhsArrayType.getShape())) {
          if (lhsDimension != ArrayType::kDynamic && rhsDimension !=
                  ArrayType::kDynamic && lhsDimension != rhsDimension) {
            // Incompatible array dimensions.
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      void rewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          assert(lhsArrayType.getRank() == rhsArrayType.getRank());

          for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
            if (lhsShape[i] == ArrayType::kDynamic ||
                rhsShape[i] == ArrayType::kDynamic) {
              mlir::Value dimensionIndex =
                  rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIndexAttr(i));

              mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getLhs(), dimensionIndex);

              mlir::Value rhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getRhs(), dimensionIndex);

              mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq,
                  lhsDimensionSize, rhsDimensionSize);

              rewriter.create<mlir::cf::AssertOp>(
                  loc, condition,
                  rewriter.getStringAttr("Incompatible dimensions"));
            }
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();

        auto dynamicDimensions =
            getArrayDynamicDimensions(rewriter, loc, op.getLhs());

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Apply the operation on each array position.
        iterateArray(
            rewriter, loc, result,
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location location,
                mlir::ValueRange indices) {
              mlir::Value lhs = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value rhs = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<SubOp>(
                  location, resultArrayType.getElementType(), lhs, rhs);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            });
      }

    private:
      bool assertions;
  };

  struct SubEWOpScalarsLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values.
      return mlir::LogicalResult::success(
          ::isScalarType(lhsType) && ::isScalarType(rhsType));
    }

    void rewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(
          op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct SubEWOpArraysLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be arrays.
      return mlir::LogicalResult::success(
          lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>());
    }

    void rewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<SubOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct SubEWOpMixedLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (lhsType.isa<ArrayType>() && ::isScalarType(rhsType)) {
        return mlir::success();
      }

      if (::isScalarType(lhsType) && rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();

      auto dynamicDimensions = getArrayDynamicDimensions(
          rewriter, loc,
          op.getLhs().getType().isa<ArrayType>()
              ? op.getLhs() : op.getRhs());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position.
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            if (op.getLhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value value = nestedBuilder.create<SubOp>(
                  location, resultArrayType.getElementType(),
                  arrayValue, op.getRhs());

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<SubOp>(
                  location, resultArrayType.getElementType(),
                  op.getLhs(), arrayValue);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });
    }
  };

  /// Product between two scalar values.
  struct MulOpLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        MulOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::MulIOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::MulFOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  /// Product between a scalar and an array.
  struct MulOpScalarProductLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      // Check if the operands are compatible.
      if (!op.getLhs().getType().isa<ArrayType>() &&
          !op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(
            op, "None of the operands is an array");
      }

      if (op.getLhs().getType().isa<ArrayType>() &&
          !::isScalarType(op.getRhs().getType())) {
        return rewriter.notifyMatchFailure(
            op, "Right-hand side operand is not a scalar");
      }

      if (op.getRhs().getType().isa<ArrayType>() &&
          !::isScalarType(op.getLhs().getType())) {
        return rewriter.notifyMatchFailure(
            op, "Left-hand side operand is not a scalar");
      }

      mlir::Value array = op.getLhs().getType().isa<ArrayType>()
          ? op.getLhs() : op.getRhs();

      mlir::Value scalar = op.getLhs().getType().isa<ArrayType>()
          ? op.getRhs() : op.getLhs();

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Multiply each array element by the scalar value.
      iterateArray(
          rewriter, loc, array,
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                location, array, indices);

            mlir::Value value = nestedBuilder.create<MulOp>(
                location, resultArrayType.getElementType(),
                scalar, arrayValue);

            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  /// Cross product of two 1-D arrays.
  /// Result is a scalar.
  ///
  /// [ x1, x2, x3 ] * [ y1, y2, y3 ] = x1 * y1 + x2 * y2 + x3 * y3
  struct MulOpCrossProductLowering : public ModelicaOpConversionPattern<MulOp>
  {
    public:
      MulOpCrossProductLowering(
        mlir::TypeConverter& typeConverter,
        mlir::MLIRContext* context,
        bool assertions)
        : ModelicaOpConversionPattern(typeConverter, context),
          assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MulOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        // Check if the operands are compatible
        if (!op.getLhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side value is not an array");
        }

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

        if (lhsArrayType.getRank() != 1) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side arrays is not 1-D");
        }

        if (!op.getRhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side value is not an array");
        }

        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (rhsArrayType.getRank() != 1) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side arrays is not 1-D");
        }

        if (lhsArrayType.getShape()[0] != ArrayType::kDynamic &&
            rhsArrayType.getShape()[0] != ArrayType::kDynamic) {
          if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
            return rewriter.notifyMatchFailure(
                op, "The two arrays have different shape");
          }
        }

        assert(lhsArrayType.getRank() == 1);
        assert(rhsArrayType.getRank() == 1);

        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          if (lhsShape[0] == ArrayType::kDynamic ||
              rhsShape[0] == ArrayType::kDynamic) {
            mlir::Value dimensionIndex =
                rewriter.create<mlir::arith::ConstantOp>(
                    loc, rewriter.getIndexAttr(0));

            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                loc, lhs, dimensionIndex);

            mlir::Value rhsDimensionSize =  rewriter.create<DimOp>(
                loc, rhs, dimensionIndex);

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                lhsDimensionSize, rhsDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }

        // Compute the result.
        mlir::Type resultType = op.getResult().getType();

        mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value upperBound = rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(0)));

        mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getZeroAttr(
                     getTypeConverter()->convertType(resultType)));

        // Iterate on the two arrays at the same time, and propagate the
        // progressive result to the next loop iteration.
        auto loop = rewriter.create<mlir::scf::ForOp>(
            loc, lowerBound, upperBound, step, init);

        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(loop.getBody());

          mlir::Value lhsScalarValue = rewriter.create<LoadOp>(
              loc, lhs, loop.getInductionVar());

          mlir::Value rhsScalarValue = rewriter.create<LoadOp>(
              loc, rhs, loop.getInductionVar());

          mlir::Value product = rewriter.create<MulOp>(
              loc, resultType, lhsScalarValue, rhsScalarValue);

          mlir::Value accumulatedValue = loop.getRegionIterArgs()[0];

          accumulatedValue = getTypeConverter()->materializeSourceConversion(
              rewriter, accumulatedValue.getLoc(),
              resultType, accumulatedValue);

          mlir::Value sum = rewriter.create<AddOp>(
              loc, resultType, product, accumulatedValue);

          sum = getTypeConverter()->materializeTargetConversion(
              rewriter, sum.getLoc(),
              getTypeConverter()->convertType(sum.getType()), sum);

          rewriter.create<mlir::scf::YieldOp>(loc, sum);
        }

        rewriter.replaceOp(op, loop.getResult(0));
        return mlir::success();
      }

    private:
      bool assertions;
  };

  /// Product of a vector (1-D array) and a matrix (2-D array).
  ///
  /// [ x1 ]  *  [ y11, y12 ]  =  [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
  /// [ x2 ]		 [ y21, y22 ]
  /// [ x3 ]		 [ y31, y32 ]
  struct MulOpVectorMatrixLowering : public ModelicaOpConversionPattern<MulOp>
  {
    public:
      MulOpVectorMatrixLowering(
        mlir::TypeConverter& typeConverter,
        mlir::MLIRContext* context,
        bool assertions)
        : ModelicaOpConversionPattern(typeConverter, context),
          assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MulOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op->getLoc();

        // Check if the operands are compatible.
        if (!op.getLhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side value is not an array");
        }

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

        if (lhsArrayType.getRank() != 1) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand size array is not 1-D");
        }

        if (!op.getRhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side value is not an array");
        }

        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (rhsArrayType.getRank() != 2) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side matrix is not 2-D");
        }

        if (lhsArrayType.getShape()[0] != ArrayType::kDynamic &&
            rhsArrayType.getShape()[0] != ArrayType::kDynamic) {
          if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
            return rewriter.notifyMatchFailure(op, "Incompatible shapes");
          }
        }

        assert(lhsArrayType.getRank() == 1);
        assert(rhsArrayType.getRank() == 2);

        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          if (lhsShape[0] == ArrayType::kDynamic || rhsShape[0] == ArrayType::kDynamic) {
            mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

            mlir::Value lhsDimensionSize =
                rewriter.create<DimOp>(loc, lhs, zero);

            mlir::Value rhsDimensionSize =
                rewriter.create<DimOp>(loc, rhs, zero);

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                lhsDimensionSize, rhsDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition,
                rewriter.getStringAttr("Incompatible dimensions"));
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();
        auto shape = resultArrayType.getShape();
        assert(shape.size() == 1);

        llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

        if (shape[0] == ArrayType::kDynamic) {
          dynamicDimensions.push_back(rewriter.create<DimOp>(
              loc, rhs, rewriter.create<mlir::arith::ConstantOp>(
                            loc, rewriter.getIndexAttr(1))));
        }

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Iterate on the columns
        mlir::Value columnsLowerBound =
            rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

        mlir::Value columnsUpperBound = rewriter.create<DimOp>(
            loc, result, rewriter.create<mlir::arith::ConstantOp>(
                             loc, rewriter.getIndexAttr(0)));

        mlir::Value columnsStep = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        auto outerLoop = rewriter.create<mlir::scf::ForOp>(
            loc, columnsLowerBound, columnsUpperBound, columnsStep);

        rewriter.setInsertionPointToStart(outerLoop.getBody());

        // Product between the vector and the current column
        mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value upperBound = rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(0)));

        mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getZeroAttr(getTypeConverter()->convertType(
                     resultArrayType.getElementType())));

        auto innerLoop = rewriter.create<mlir::scf::ForOp>(
            loc, lowerBound, upperBound, step, init);

        rewriter.setInsertionPointToStart(innerLoop.getBody());

        mlir::Value lhsScalarValue = rewriter.create<LoadOp>(
            loc, lhs, innerLoop.getInductionVar());

        mlir::Value rhsScalarValue = rewriter.create<LoadOp>(
            loc, rhs, mlir::ValueRange({
                          innerLoop.getInductionVar(),
                          outerLoop.getInductionVar()
                      }));

        mlir::Value product = rewriter.create<MulOp>(
            loc, resultArrayType.getElementType(),
            lhsScalarValue, rhsScalarValue);

        mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

        accumulatedValue = getTypeConverter()->materializeSourceConversion(
            rewriter, accumulatedValue.getLoc(),
            resultArrayType.getElementType(), accumulatedValue);

        mlir::Value sum = rewriter.create<AddOp>(
            loc, resultArrayType.getElementType(), product, accumulatedValue);

        sum = getTypeConverter()->materializeTargetConversion(
            rewriter, sum.getLoc(),
            getTypeConverter()->convertType(sum.getType()), sum);

        rewriter.create<mlir::scf::YieldOp>(loc, sum);

        // Store the product in the result array.
        rewriter.setInsertionPointAfter(innerLoop);
        mlir::Value productResult = innerLoop.getResult(0);

        productResult = getTypeConverter()->materializeSourceConversion(
            rewriter, productResult.getLoc(),
            resultArrayType.getElementType(), productResult);

        rewriter.create<StoreOp>(
            loc, productResult, result, outerLoop.getInductionVar());

        rewriter.setInsertionPointAfter(outerLoop);
        return mlir::success();
      }

    private:
      bool assertions;
  };

  /// Product of a matrix (2-D array) and a vector (1-D array).
  ///
  /// [ x11, x12 ] * [ y1, y2 ] = [ x11 * y1 + x12 * y2 ]
  /// [ x21, x22 ]							  [ x21 * y1 + x22 * y2 ]
  /// [ x31, x32 ]								[ x31 * y1 + x22 * y2 ]
  struct MulOpMatrixVectorLowering : public ModelicaOpConversionPattern<MulOp>
  {
    public:
      MulOpMatrixVectorLowering(
        mlir::TypeConverter& typeConverter,
        mlir::MLIRContext* context,
        bool assertions)
        : ModelicaOpConversionPattern(typeConverter, context),
          assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MulOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op->getLoc();

        // Check if the operands are compatible.
        if (!op.getLhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side value is not an array");
        }

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

        if (lhsArrayType.getRank() != 2) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand size array is not 2-D");
        }

        if (!op.getRhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side value is not an array");
        }

        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (rhsArrayType.getRank() != 1) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side matrix is not 1-D");
        }

        if (lhsArrayType.getShape()[1] != ArrayType::kDynamic &&
            rhsArrayType.getShape()[0] != ArrayType::kDynamic) {
          if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
            return rewriter.notifyMatchFailure(op, "Incompatible shapes");
          }
        }

        assert(lhsArrayType.getRank() == 2);
        assert(rhsArrayType.getRank() == 1);

        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          if (lhsShape[1] == ArrayType::kDynamic ||
              rhsShape[0] == ArrayType::kDynamic) {
            mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

            mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(1));

            mlir::Value lhsDimensionSize =
                rewriter.create<DimOp>(loc, lhs, one);

            mlir::Value rhsDimensionSize =
                rewriter.create<DimOp>(loc, rhs, zero);

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                lhsDimensionSize, rhsDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition,
                rewriter.getStringAttr("Incompatible dimensions"));
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();
        auto shape = resultArrayType.getShape();

        llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

        if (shape[0] == ArrayType::kDynamic) {
          dynamicDimensions.push_back(rewriter.create<DimOp>(
              loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                            loc, rewriter.getIndexAttr(0))));
        }

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Iterate on the rows.
        mlir::Value rowsLowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value rowsUpperBound = rewriter.create<DimOp>(
            loc, result, rewriter.create<mlir::arith::ConstantOp>(
                             loc, rewriter.getIndexAttr(0)));

        mlir::Value rowsStep = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        auto outerLoop = rewriter.create<mlir::scf::ForOp>(
            loc, rowsLowerBound, rowsUpperBound, rowsStep);

        rewriter.setInsertionPointToStart(outerLoop.getBody());

        // Product between the current row and the vector.
        mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value upperBound = rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(1)));

        mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getZeroAttr(getTypeConverter()->convertType(
                     resultArrayType.getElementType())));

        auto innerLoop = rewriter.create<mlir::scf::ForOp>(
            loc, lowerBound, upperBound, step, init);

        rewriter.setInsertionPointToStart(innerLoop.getBody());

        mlir::Value lhsScalarValue = rewriter.create<LoadOp>(
            loc, lhs, mlir::ValueRange({
                          outerLoop.getInductionVar(),
                          innerLoop.getInductionVar()
                      }));

        mlir::Value rhsScalarValue = rewriter.create<LoadOp>(
            loc, rhs, innerLoop.getInductionVar());

        mlir::Value product = rewriter.create<MulOp>(
            loc, resultArrayType.getElementType(),
            lhsScalarValue, rhsScalarValue);

        mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

        accumulatedValue = getTypeConverter()->materializeSourceConversion(
            rewriter, accumulatedValue.getLoc(),
            resultArrayType.getElementType(), accumulatedValue);

        mlir::Value sum = rewriter.create<AddOp>(
            loc, resultArrayType.getElementType(), product, accumulatedValue);

        sum = getTypeConverter()->materializeTargetConversion(
            rewriter, sum.getLoc(),
            getTypeConverter()->convertType(sum.getType()), sum);

        rewriter.create<mlir::scf::YieldOp>(loc, sum);

        // Store the product in the result array.
        rewriter.setInsertionPointAfter(innerLoop);
        mlir::Value productResult = innerLoop.getResult(0);

        productResult = getTypeConverter()->materializeSourceConversion(
            rewriter, productResult.getLoc(),
            resultArrayType.getElementType(), productResult);

        rewriter.create<StoreOp>(
            loc, productResult, result, outerLoop.getInductionVar());

        rewriter.setInsertionPointAfter(outerLoop);

        return mlir::success();
      }

    private:
      bool assertions;
  };

  /// Product of two matrices (2-D arrays).
  ///
  /// [ x11, x12, x13 ] * [ y11, y12 ]  =  [ x11 * y11 + x12 * y21 + x13 * y31, x11 * y12 + x12 * y22 + x13 * y32 ]
  /// [ x21, x22, x23 ]   [ y21, y22 ]		 [ x21 * y11 + x22 * y21 + x23 * y31, x21 * y12 + x22 * y22 + x23 * y32 ]
  /// [ x31, x32, x33 ]	  [ y31, y32 ]		 [ x31 * y11 + x32 * y21 + x33 * y31, x31 * y12 + x32 * y22 + x33 * y32 ]
  /// [ x41, x42, x43 ]
  struct MulOpMatrixLowering : public ModelicaOpConversionPattern<MulOp>
  {
    public:
      MulOpMatrixLowering(
        mlir::TypeConverter& typeConverter,
        mlir::MLIRContext* context,
        bool assertions)
        : ModelicaOpConversionPattern(typeConverter, context),
          assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          MulOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        // Check if the operands are compatible.
        if (!op.getLhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand side value is not an array");
        }

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

        if (lhsArrayType.getRank() != 2) {
          return rewriter.notifyMatchFailure(
              op, "Left-hand size array is not 2-D");
        }

        if (!op.getRhs().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side value is not an array");
        }

        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (rhsArrayType.getRank() != 2) {
          return rewriter.notifyMatchFailure(
              op, "Right-hand side matrix is not 2-D");
        }

        if (lhsArrayType.getShape()[1] != ArrayType::kDynamic &&
            rhsArrayType.getShape()[0] != ArrayType::kDynamic) {
          if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
            return rewriter.notifyMatchFailure(op, "Incompatible shapes");
          }
        }

        assert(lhsArrayType.getRank() == 2);
        assert(rhsArrayType.getRank() == 2);

        mlir::Value lhs = op.getLhs();
        mlir::Value rhs = op.getRhs();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          if (lhsShape[1] == ArrayType::kDynamic ||
              rhsShape[0] == ArrayType::kDynamic) {
            mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

            mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(1));

            mlir::Value lhsDimensionSize =
                rewriter.create<DimOp>(loc, lhs, one);

            mlir::Value rhsDimensionSize =
                rewriter.create<DimOp>(loc, rhs, zero);

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                lhsDimensionSize, rhsDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition,
                rewriter.getStringAttr("Incompatible dimensions"));
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();
        auto shape = resultArrayType.getShape();

        llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

        if (shape[0] == ArrayType::kDynamic) {
          dynamicDimensions.push_back(rewriter.create<DimOp>(
              loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                            loc, rewriter.getIndexAttr(0))));
        }

        if (shape[1] == ArrayType::kDynamic) {
          dynamicDimensions.push_back(rewriter.create<DimOp>(
              loc, rhs, rewriter.create<mlir::arith::ConstantOp>(
                            loc, rewriter.getIndexAttr(1))));
        }

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Iterate on the rows.
        mlir::Value rowsLowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value rowsUpperBound = rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(0)));

        mlir::Value rowsStep = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        auto rowsLoop = rewriter.create<mlir::scf::ForOp>(
            loc, rowsLowerBound, rowsUpperBound, rowsStep);

        rewriter.setInsertionPointToStart(rowsLoop.getBody());

        // Iterate on the columns.
        mlir::Value columnsLowerBound =
            rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

        mlir::Value columnsUpperBound = rewriter.create<DimOp>(
            loc, rhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(1)));

        mlir::Value columnsStep = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        auto columnsLoop = rewriter.create<mlir::scf::ForOp>(
            loc, columnsLowerBound, columnsUpperBound, columnsStep);

        rewriter.setInsertionPointToStart(columnsLoop.getBody());

        // Product between the current row and the current column.
        mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value upperBound = rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(
                          loc, rewriter.getIndexAttr(1)));

        mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getZeroAttr(getTypeConverter()->convertType(
                     resultArrayType.getElementType())));

        auto innerLoop = rewriter.create<mlir::scf::ForOp>(
            loc, lowerBound, upperBound, step, init);

        rewriter.setInsertionPointToStart(innerLoop.getBody());

        mlir::Value lhsScalarValue = rewriter.create<LoadOp>(
            loc, lhs, mlir::ValueRange({
                          rowsLoop.getInductionVar(),
                          innerLoop.getInductionVar()
                      }));

        mlir::Value rhsScalarValue = rewriter.create<LoadOp>(
            loc, rhs, mlir::ValueRange({
                          innerLoop.getInductionVar(),
                          columnsLoop.getInductionVar()
                      }));

        mlir::Value product = rewriter.create<MulOp>(
            loc, resultArrayType.getElementType(),
            lhsScalarValue, rhsScalarValue);

        mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

        accumulatedValue = getTypeConverter()->materializeSourceConversion(
            rewriter, accumulatedValue.getLoc(),
            resultArrayType.getElementType(), accumulatedValue);

        mlir::Value sum = rewriter.create<AddOp>(
            loc, resultArrayType.getElementType(), product, accumulatedValue);

        sum = getTypeConverter()->materializeTargetConversion(
            rewriter, sum.getLoc(),
            getTypeConverter()->convertType(sum.getType()), sum);

        rewriter.create<mlir::scf::YieldOp>(loc, sum);

        // Store the product in the result array.
        rewriter.setInsertionPointAfter(innerLoop);
        mlir::Value productResult = innerLoop.getResult(0);

        productResult = getTypeConverter()->materializeSourceConversion(
            rewriter, productResult.getLoc(), resultArrayType.getElementType(), productResult);

        rewriter.create<StoreOp>(
            loc, productResult, result, mlir::ValueRange({
                                            rowsLoop.getInductionVar(),
                                            columnsLoop.getInductionVar()
                                        }));

        rewriter.setInsertionPointAfter(rowsLoop);

        return mlir::success();
      }

    private:
      bool assertions;
  };

  /// Element-wise product of two scalar values.
  struct MulEWOpScalarsLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MulEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values.
      return mlir::LogicalResult::success(
          ::isScalarType(lhsType) && ::isScalarType(rhsType));
    }

    void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  /// Element-wise product of two arrays.
  struct MulEWOpArraysLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    public:
      MulEWOpArraysLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern(context), assertions(assertions)
      {
      }

      mlir::LogicalResult match(MulEWOp op) const override
      {
        mlir::Type lhsType = op.getLhs().getType();
        mlir::Type rhsType = op.getRhs().getType();

        if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
          return mlir::failure();
        }

        auto lhsArrayType = lhsType.cast<ArrayType>();
        auto rhsArrayType = rhsType.cast<ArrayType>();

        if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
          // Incompatible ranks.
          return mlir::failure();
        }

        for (const auto& [lhsDimension, rhsDimension] : llvm::zip(
                 lhsArrayType.getShape(), rhsArrayType.getShape())) {
          if (lhsDimension != ArrayType::kDynamic &&
              rhsDimension != ArrayType::kDynamic &&
              lhsDimension != rhsDimension) {
            // Incompatible array dimensions.
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          assert(lhsArrayType.getRank() == rhsArrayType.getRank());

          for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
            if (lhsShape[i] == ArrayType::kDynamic ||
                rhsShape[i] == ArrayType::kDynamic) {
              mlir::Value dimensionIndex =
                  rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIndexAttr(i));

              mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getLhs(), dimensionIndex);

              mlir::Value rhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getRhs(), dimensionIndex);

              mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq,
                  lhsDimensionSize, rhsDimensionSize);

              rewriter.create<mlir::cf::AssertOp>(
                  loc, condition,
                  rewriter.getStringAttr("Incompatible dimensions"));
            }
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();

        auto dynamicDimensions = getArrayDynamicDimensions(
            rewriter, loc, op.getLhs());

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Apply the operation on each array position.
        iterateArray(
            rewriter, loc, op.getLhs(),
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location location,
                mlir::ValueRange indices) {
              mlir::Value lhs = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value rhs = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<MulOp>(
                  location, resultArrayType.getElementType(), lhs, rhs);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            });
      }

    private:
      bool assertions;
  };

  /// Element-wise product between a scalar value and an array.
  struct MulEWOpMixedLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MulEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (lhsType.isa<ArrayType>() && ::isScalarType(rhsType)) {
        return mlir::success();
      }

      if (::isScalarType(lhsType) && rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<MulOp>(
          op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  /// Division between two scalar values.
  struct DivOpLowering : public ModelicaOpConversionPattern<DivOp>
  {
    using ModelicaOpConversionPattern<DivOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        DivOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      mlir::Type lhsType = lhs.getType();
      mlir::Type rhsType = rhs.getType();

      if (!lhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported left-hand side operand type");
      }

      if (!rhsType.isa<
              mlir::IndexType, mlir::IntegerType, mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported right-hand side operand type");
      }

      std::tie(lhs, rhs) = castToMostGenericType(rewriter, lhs, rhs);

      mlir::Location loc = op.getLoc();
      mlir::Type operandsType = lhs.getType();

      if (operandsType.isa<mlir::IndexType, mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::DivSIOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::DivFOp>(
            loc, operandsType, lhs, rhs);

        result = castToType(
            rewriter, result,
            getTypeConverter()->convertType(op.getResult().getType()));

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  /// Division between a an array and a scalar value (but not vice versa).
  struct DivOpMixedLowering : public ModelicaOpRewritePattern<DivOp>
  {
    using ModelicaOpRewritePattern<DivOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          lhsType.isa<ArrayType>() && ::isScalarType(rhsType));
    }

    void rewrite(DivOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();

      auto dynamicDimensions = getArrayDynamicDimensions(
          rewriter, loc, op.getLhs().getType().isa<ArrayType>()
              ? op.getLhs() : op.getRhs());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position.
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                location, op.getLhs(), indices);

            mlir::Value value = nestedBuilder.create<DivOp>(
                location, resultArrayType.getElementType(),
                arrayValue, op.getRhs());

            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  /// Element-wise division of two scalar values.
  struct DivEWOpScalarsLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    using ModelicaOpRewritePattern<DivEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values
      return mlir::LogicalResult::success(
          ::isScalarType(lhsType) && ::isScalarType(rhsType));
    }

    void rewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<DivOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  /// Element-wise division between two arrays.
  struct DivEWOpArraysLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    public:
      DivEWOpArraysLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern(context), assertions(assertions)
      {
      }

      mlir::LogicalResult match(DivEWOp op) const override
      {
        mlir::Type lhsType = op.getLhs().getType();
        mlir::Type rhsType = op.getRhs().getType();

        if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
          return mlir::failure();
        }

        auto lhsArrayType = lhsType.cast<ArrayType>();
        auto rhsArrayType = rhsType.cast<ArrayType>();

        if (lhsArrayType.getRank() != rhsArrayType.getRank()) {
          // Incompatible ranks.
          return mlir::failure();
        }

        for (const auto& [lhsDimension, rhsDimension] : llvm::zip(
                 lhsArrayType.getShape(), rhsArrayType.getShape())) {
          if (lhsDimension != ArrayType::kDynamic &&
              rhsDimension != ArrayType::kDynamic &&
              lhsDimension != rhsDimension) {
            // Incompatible array dimensions.
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      void rewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
        auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

        if (assertions) {
          // Check if the dimensions are compatible.
          auto lhsShape = lhsArrayType.getShape();
          auto rhsShape = rhsArrayType.getShape();

          assert(lhsArrayType.getRank() == rhsArrayType.getRank());

          for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
            if (lhsShape[i] == ArrayType::kDynamic ||
                rhsShape[i] == ArrayType::kDynamic) {
              mlir::Value dimensionIndex =
                  rewriter.create<mlir::arith::ConstantOp>(
                      loc, rewriter.getIndexAttr(i));

              mlir::Value lhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getLhs(), dimensionIndex);

              mlir::Value rhsDimensionSize = rewriter.create<DimOp>(
                  loc, op.getRhs(), dimensionIndex);

              mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq,
                  lhsDimensionSize, rhsDimensionSize);

              rewriter.create<mlir::cf::AssertOp>(
                  loc, condition,
                  rewriter.getStringAttr("Incompatible dimensions"));
            }
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();

        auto dynamicDimensions =
            getArrayDynamicDimensions(rewriter, loc, op.getLhs());

        mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
            op, resultArrayType, dynamicDimensions);

        // Apply the operation on each array position.
        iterateArray(
            rewriter, loc, op.getLhs(),
            [&](mlir::OpBuilder& nestedBuilder,
                mlir::Location location,
                mlir::ValueRange indices) {
              mlir::Value lhs = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value rhs = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<DivOp>(
                  location, resultArrayType.getElementType(), lhs, rhs);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            });
      }

    private:
      bool assertions;
  };

  /// Element-wise division between a scalar value and an array (and vice
  /// versa).
  struct DivEWOpMixedLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    using ModelicaOpRewritePattern<DivEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (lhsType.isa<ArrayType>() && !rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      if (!lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      // Allocate the result array.
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();

      auto dynamicDimensions = getArrayDynamicDimensions(
          rewriter, loc, op.getLhs().getType().isa<ArrayType>()
              ? op.getLhs() : op.getRhs());

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(
          op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position.
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder,
              mlir::Location location,
              mlir::ValueRange indices) {
            if (op.getLhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                  location, op.getLhs(), indices);

              mlir::Value value = nestedBuilder.create<DivOp>(
                  location, resultArrayType.getElementType(),
                  arrayValue, op.getRhs());

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(
                  location, op.getRhs(), indices);

              mlir::Value value = nestedBuilder.create<DivOp>(
                  location, resultArrayType.getElementType(),
                  op.getLhs(), arrayValue);

              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });
    }
  };

  struct PowOpMatrixLowering : public ModelicaOpRewritePattern<PowOp>
  {
    public:
      PowOpMatrixLowering(mlir::MLIRContext* context, bool assertions)
          : ModelicaOpRewritePattern(context), assertions(assertions)
      {
      }

      mlir::LogicalResult matchAndRewrite(PowOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op->getLoc();

        // Check if the operands are compatible.
        if (!op.getBase().getType().isa<ArrayType>()) {
          return rewriter.notifyMatchFailure(op, "Base is not an array");
        }

        auto baseArrayType = op.getBase().getType().cast<ArrayType>();

        if (baseArrayType.getRank() != 2) {
          return rewriter.notifyMatchFailure(op, "Base array is not 2-D");
        }

        if (baseArrayType.getShape()[0] != ArrayType::kDynamic &&
            baseArrayType.getShape()[1] != ArrayType::kDynamic) {
          if (baseArrayType.getShape()[0] != baseArrayType.getShape()[1]) {
            return rewriter.notifyMatchFailure(
                op, "Base is not a square matrix");
          }
        }

        assert(baseArrayType.getRank() == 2);

        if (assertions) {
          // Check if the matrix is a square one.
          auto shape = baseArrayType.getShape();

          if (shape[0] == ArrayType::kDynamic ||
              shape[1] == ArrayType::kDynamic) {
            mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(0));

            mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(1));

            mlir::Value firstDimensionSize = rewriter.create<DimOp>(
                loc, op.getBase(), zero);

            mlir::Value secondDimensionSize = rewriter.create<DimOp>(
                loc, op.getBase(), one);

            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::eq,
                firstDimensionSize, secondDimensionSize);

            rewriter.create<mlir::cf::AssertOp>(
                loc, condition,
                rewriter.getStringAttr("Base matrix is not squared"));
          }
        }

        // Allocate the result array.
        auto resultArrayType = op.getResult().getType().cast<ArrayType>();

        auto dynamicDimensions =
            getArrayDynamicDimensions(rewriter, loc, op.getBase());

        mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value size = rewriter.create<DimOp>(loc, op.getBase(), one);

        mlir::Value result = rewriter.replaceOpWithNewOp<IdentityOp>(
            op, resultArrayType, size);

        // Compute the result.
        mlir::Value exponent = rewriter.create<CastOp>(
            loc, rewriter.getIndexType(), op.getExponent());

        exponent = rewriter.create<mlir::arith::AddIOp>(
            loc, rewriter.getIndexType(), exponent,
            rewriter.create<mlir::arith::ConstantOp>(
                loc, rewriter.getIndexAttr(1)));

        mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        auto forLoop = rewriter.create<mlir::scf::ForOp>(
            loc, lowerBound, exponent, step);

        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(forLoop.getBody());

          mlir::Value next = rewriter.create<MulOp>(
              loc, result.getType(), result, op.getBase());

          rewriter.create<AssignmentOp>(loc, result, next);
        }

        return mlir::success();
      }

    private:
      bool assertions;
  };

  struct ReductionOpLowering : public ModelicaOpConversionPattern<ReductionOp>
  {
      using ModelicaOpConversionPattern<ReductionOp>
          ::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          ReductionOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        if (llvm::any_of(op.getIterables(), [](mlir::Value iterable) {
              return !iterable.getType().isa<RangeType>();
            })) {
          return mlir::failure();
        }

        // Determine the initial value of the accumulator.
        auto yieldOp = mlir::cast<YieldOp>(op.getBody()->getTerminator());

        mlir::Type resultType =
            getTypeConverter()->convertType(yieldOp.getValues()[0].getType());

        llvm::SmallVector<mlir::Value, 1> initialValues;

        if (mlir::failed(getInitialValues(
                rewriter, loc, op.getAction(), resultType, initialValues))) {
          return mlir::failure();
        }

        // Compute the boundaries and steps so that the step becomes positive.
        llvm::SmallVector<mlir::Value, 6> lowerBounds;
        llvm::SmallVector<mlir::Value, 6> upperBounds;
        llvm::SmallVector<mlir::Value, 6> steps;

        mlir::Value zero = rewriter.create<ConstantOp>(
            loc, rewriter.getIndexAttr(0));

        mlir::Value one = rewriter.create<ConstantOp>(
            loc, rewriter.getIndexAttr(1));

        mlir::Type indexType = rewriter.getIndexType();

        for (mlir::Value iterable : op.getIterables()) {
          mlir::Value step = rewriter.create<RangeStepOp>(loc, iterable);

          mlir::Value isNonNegativeStep = rewriter.create<GteOp>(
              loc, BooleanType::get(rewriter.getContext()), step, zero);

          mlir::Value beginValue =
              rewriter.create<RangeBeginOp>(loc, iterable);

          mlir::Value size = rewriter.create<RangeSizeOp>(loc, iterable);

          // begin + (size - 1) * step
          mlir::Value actualEndValue = rewriter.create<AddOp>(
              loc, indexType, beginValue,
              rewriter.create<MulOp>(
                  loc, indexType,
                  step, rewriter.create<SubOp>(loc, indexType, size, one)));

          auto lowerBoundSelectOp = rewriter.create<SelectOp>(
              loc, indexType, isNonNegativeStep, beginValue, actualEndValue);

          auto upperBoundSelectOp = rewriter.create<SelectOp>(
              loc, indexType, isNonNegativeStep, actualEndValue, beginValue);

          mlir::Value negatedStep =
              rewriter.create<NegateOp>(loc, indexType, step);

          auto stepSelectOp = rewriter.create<SelectOp>(
              loc, indexType, isNonNegativeStep, step, negatedStep);

          lowerBounds.push_back(lowerBoundSelectOp.getResult(0));

          mlir::Value upperBound = upperBoundSelectOp.getResult(0);

          upperBound = rewriter.create<AddOp>(
              loc, upperBound.getType(), upperBound, one);

          upperBounds.push_back(upperBound);
          steps.push_back(stepSelectOp.getResult(0));
        }

        // Create the parallel loops.
        auto parallelOp = rewriter.create<mlir::scf::ParallelOp>(
            loc, lowerBounds, upperBounds, steps, initialValues);

        // Erase the default-created YieldOp inside ParallelOp.
        rewriter.eraseOp(parallelOp.getBody()->getTerminator());

        // Create the reduce operation.
        rewriter.setInsertionPoint(yieldOp);

        mlir::Value currentElementValue = yieldOp.getValues()[0];

        currentElementValue = getTypeConverter()->materializeTargetConversion(
            rewriter, loc, resultType, currentElementValue);

        auto reduceOp = rewriter.create<mlir::scf::ReduceOp>(
            loc, currentElementValue);

        rewriter.setInsertionPointToEnd(
            &reduceOp.getReductionOperator().front());

        mlir::Value reductionResult = computeReductionResult(
            rewriter, loc, op.getAction(), resultType,
            reduceOp.getReductionOperator().front().getArgument(0),
            reduceOp.getReductionOperator().front().getArgument(1));

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

        // Recreate the YieldOp for ParallelOp.
        rewriter.setInsertionPointToEnd(parallelOp.getBody());
        rewriter.create<mlir::scf::YieldOp>(loc);

        rewriter.replaceOp(op, parallelOp);

        return mlir::success();
      }

    private:
      mlir::LogicalResult getInitialValues(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef action,
          mlir::Type resultType,
          llvm::SmallVectorImpl<mlir::Value>& initialValues) const
      {
        if (action == "add") {
          if (resultType.isa<mlir::IntegerType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIntegerAttr(resultType, 0)));

            return mlir::success();
          }

          if (resultType.isa<mlir::FloatType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getFloatAttr(resultType, 0)));

            return mlir::success();
          }

          if (resultType.isa<mlir::IndexType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIndexAttr(0)));

            return mlir::success();
          }
        }

        if (action == "mul") {
          if (resultType.isa<mlir::IntegerType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIntegerAttr(resultType, 1)));

            return mlir::success();
          }

          if (resultType.isa<mlir::FloatType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getFloatAttr(resultType, 1)));

            return mlir::success();
          }

          if (resultType.isa<mlir::IndexType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIndexAttr(1)));

            return mlir::success();
          }
        }

        if (action == "min") {
          if (resultType.isa<mlir::IntegerType>()) {
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

          if (resultType.isa<mlir::FloatType>()) {
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

          if (resultType.isa<mlir::IndexType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIndexAttr(std::numeric_limits<int64_t>::max())));

            return mlir::success();
          }
        }

        if (action == "max") {
          if (resultType.isa<mlir::IntegerType>()) {
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

          if (resultType.isa<mlir::FloatType>()) {
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

          if (resultType.isa<mlir::IndexType>()) {
            initialValues.push_back(builder.create<ConstantOp>(
                loc, builder.getIndexAttr(std::numeric_limits<int64_t>::min())));

            return mlir::success();
          }
        }

        return mlir::failure();
      }

      mlir::Value computeReductionResult(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef action,
          mlir::Type resultType,
          mlir::Value lhs,
          mlir::Value rhs) const
      {
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
}

//===---------------------------------------------------------------------===//
// Various operations
//===---------------------------------------------------------------------===//

namespace
{
  struct SelectOpLowering : public ModelicaOpConversionPattern<SelectOp>
  {
    using ModelicaOpConversionPattern<SelectOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        SelectOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getTrueValues().size() > 1 ||
          adaptor.getFalseValues().size() > 1) {
        return rewriter.notifyMatchFailure(
            op, "Multiple values among true or false cases");
      }

      assert(adaptor.getTrueValues().size() ==
             adaptor.getFalseValues().size());

      if (!::isScalarType(adaptor.getTrueValues()[0].getType()) ||
          !::isScalarType(adaptor.getFalseValues()[0].getType())) {
        return rewriter.notifyMatchFailure(
            op, "Unsupported operand types");
      }

      mlir::Location loc = op.getLoc();

      mlir::Value condition = castToInteger(
          rewriter, adaptor.getCondition(), rewriter.getI1Type());

      mlir::Value trueValue = adaptor.getTrueValues()[0];
      mlir::Value falseValue = adaptor.getFalseValues()[0];

      std::tie(trueValue, falseValue) = castToMostGenericType(
          rewriter, trueValue, falseValue);

      mlir::Value result = rewriter.create<mlir::arith::SelectOp>(
          loc, condition, trueValue, falseValue);

      result = castToType(
          rewriter, result,
          getTypeConverter()->convertType(op.getResultTypes()[0]));

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct SelectOpMultipleValuesPattern
      : public ModelicaOpRewritePattern<SelectOp>
  {
    using ModelicaOpRewritePattern<SelectOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SelectOp op) const override
    {
      return mlir::LogicalResult::success(
          op.getTrueValues().size() > 1 || op.getFalseValues().size() > 1);
    }

    void rewrite(
        SelectOp op,
        mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();

      mlir::TypeRange resultTypes = op.getResultTypes();
      llvm::SmallVector<mlir::Value> results;

      assert(resultTypes.size() == op.getTrueValues().size());
      assert(resultTypes.size() == op.getFalseValues().size());

      for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
        mlir::Value trueValue = op.getTrueValues()[i];
        mlir::Value falseValue = op.getFalseValues()[i];

        auto resultOp = rewriter.create<SelectOp>(
            loc, resultTypes[i],
            op.getCondition(), trueValue, falseValue);

        results.push_back(resultOp.getResult(0));
      }

      rewriter.replaceOp(op, results);
    }
  };
}

static void populateModelicaToArithPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::TypeConverter& typeConverter,
    bool assertions)
{
  // Range operations.
  patterns.insert<RangeSizeOpLowering>(context);

  // Comparison operations.
  patterns.insert<
      EqOpLowering,
      NotEqOpLowering,
      GtOpLowering,
      GteOpLowering,
      LtOpLowering,
      LteOpLowering>(typeConverter, context);

  // Logic operations.
  patterns.insert<
      NotOpArrayLowering>(context);

  patterns.insert<
      AndOpArrayLowering,
      OrOpArrayLowering>(context, assertions);

  patterns.insert<
      NotOpIntegerLowering,
      NotOpFloatLowering,
      AndOpIntegersLikeLowering,
      AndOpFloatsLowering,
      AndOpIntegerLikeFloatLowering,
      AndOpFloatIntegerLikeLowering,
      OrOpIntegersLikeLowering,
      OrOpFloatsLowering,
      OrOpIntegerLikeFloatLowering,
      OrOpFloatIntegerLikeLowering>(typeConverter, context);

  // Math operations.
  patterns.insert<
      NegateOpArrayLowering>(context);

  patterns.insert<
      AddOpArraysLowering>(context, assertions);

  patterns.insert<
      AddEWOpScalarsLowering,
      AddEWOpArraysLowering,
      AddEWOpMixedLowering>(context);

  patterns.insert<
      SubOpArraysLowering>(context, assertions);

  patterns.insert<
      SubEWOpScalarsLowering,
      SubEWOpArraysLowering,
      SubEWOpMixedLowering,
      MulOpScalarProductLowering,
      MulEWOpScalarsLowering>(context);

  patterns.insert<
      MulEWOpArraysLowering>(context, assertions);

  patterns.insert<
      MulEWOpMixedLowering,
      DivOpMixedLowering,
      DivEWOpScalarsLowering>(context);

  patterns.insert<
      DivEWOpArraysLowering>(context, assertions);

  patterns.insert<
      DivEWOpMixedLowering>(context);

  patterns.insert<
      PowOpMatrixLowering>(context, assertions);

  patterns.insert<
      ConstantOpScalarLowering,
      NegateOpLowering,
      AddOpLowering,
      SubOpLowering,
      MulOpLowering,
      DivOpLowering>(typeConverter, context);

  patterns.insert<
      MulOpCrossProductLowering,
      MulOpVectorMatrixLowering,
      MulOpMatrixVectorLowering,
      MulOpMatrixLowering>(typeConverter, context, assertions);

  patterns.insert<ReductionOpLowering>(typeConverter, context);

  // Various operations.
  patterns.insert<
      SelectOpLowering>(typeConverter, context);

  patterns.insert<
      SelectOpMultipleValuesPattern>(context);
}

namespace
{
  class ModelicaToArithConversionPass
      : public mlir::impl::ModelicaToArithConversionPassBase<
          ModelicaToArithConversionPass>
  {
    public:
      using ModelicaToArithConversionPassBase<ModelicaToArithConversionPass>
          ::ModelicaToArithConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<RangeSizeOp>();

        target.addIllegalOp<
            EqOp,
            NotEqOp,
            GtOp,
            GteOp,
            LtOp,
            LteOp>();

        target.addIllegalOp<
            NotOp,
            AndOp,
            OrOp>();

        target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
          mlir::Type resultType = op.getResult().getType();
          return mlir::isa<ArrayType, RangeType>(resultType);
        });

        target.addIllegalOp<
            NegateOp,
            AddOp,
            AddEWOp,
            SubOp,
            SubEWOp,
            MulOp,
            MulEWOp,
            DivOp,
            DivEWOp>();

        target.addDynamicallyLegalOp<PowOp>([](PowOp op) {
          return !op.getBase().getType().isa<ArrayType>();
        });

        target.addIllegalOp<ReductionOp>();
        target.addIllegalOp<SelectOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        TypeConverter typeConverter(bitWidth);
        mlir::RewritePatternSet patterns(&getContext());

        populateModelicaToArithPatterns(
            patterns, &getContext(), typeConverter, assertions);

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToArithConversionPass()
  {
    return std::make_unique<ModelicaToArithConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToArithConversionPass(
      const ModelicaToArithConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToArithConversionPass>(options);
  }
}
