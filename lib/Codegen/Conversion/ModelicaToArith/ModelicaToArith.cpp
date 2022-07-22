#include "marco/Codegen/Conversion/ModelicaToArith/ModelicaToArith.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static void iterateArray(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value array,
    std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)> callback)
{
  assert(array.getType().isa<ArrayType>());
  auto arrayType = array.getType().cast<ArrayType>();

  mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  mlir::Value one = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));

  llvm::SmallVector<mlir::Value, 3> lowerBounds(arrayType.getRank(), zero);
  llvm::SmallVector<mlir::Value, 3> upperBounds;
  llvm::SmallVector<mlir::Value, 3> steps(arrayType.getRank(), one);

  for (unsigned int i = 0, e = arrayType.getRank(); i < e; ++i) {
    mlir::Value dim = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(i));
    upperBounds.push_back(builder.create<DimOp>(loc, array, dim));
  }

  // Create nested loops in order to iterate on each dimension of the array
  mlir::scf::buildLoopNest(builder, loc, lowerBounds, upperBounds, steps, callback);
}

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      ModelicaOpRewritePattern(mlir::MLIRContext* ctx, ModelicaToArithOptions options)
          : mlir::OpRewritePattern<Op>(ctx),
            options(std::move(options))
      {
      }

    protected:
      ModelicaToArithOptions options;
  };

  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      ModelicaOpConversionPattern(mlir::MLIRContext* ctx, mlir::LLVMTypeConverter& typeConverter, ModelicaToArithOptions options)
          : mlir::ConvertOpToLLVMPattern<Op>(typeConverter, 1),
            options(std::move(options))
      {
      }

    protected:
      ModelicaToArithOptions options;
  };
}

//===----------------------------------------------------------------------===//
// Comparison operations
//===----------------------------------------------------------------------===//

namespace
{
  struct EqOpCastPattern : public ModelicaOpRewritePattern<EqOp>
  {
    using ModelicaOpRewritePattern<EqOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(EqOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<EqOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct EqOpLowering : public ModelicaOpConversionPattern<EqOp>
  {
    using ModelicaOpConversionPattern<EqOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(EqOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::eq, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OEQ, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct NotEqOpCastPattern : public ModelicaOpRewritePattern<NotEqOp>
  {
    using ModelicaOpRewritePattern<NotEqOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NotEqOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<NotEqOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct NotEqOpLowering : public ModelicaOpConversionPattern<NotEqOp>
  {
    using ModelicaOpConversionPattern<NotEqOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(NotEqOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::ne, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct GtOpCastPattern : public ModelicaOpRewritePattern<GtOp>
  {
    using ModelicaOpRewritePattern<GtOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(GtOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<GtOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct GtOpLowering : public ModelicaOpConversionPattern<GtOp>
  {
    using ModelicaOpConversionPattern<GtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(GtOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sgt, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGT, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct GteOpCastPattern : public ModelicaOpRewritePattern<GteOp>
  {
    using ModelicaOpRewritePattern<GteOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(GteOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<GteOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct GteOpLowering : public ModelicaOpConversionPattern<GteOp>
  {
    using ModelicaOpConversionPattern<GteOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(GteOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sge, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct LtOpCastPattern : public ModelicaOpRewritePattern<LtOp>
  {
    using ModelicaOpRewritePattern<LtOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(LtOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<LtOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct LtOpLowering : public ModelicaOpConversionPattern<LtOp>
  {
    using ModelicaOpConversionPattern<LtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LtOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::slt, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLT, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };

  struct LteOpCastPattern : public ModelicaOpRewritePattern<LteOp>
  {
    using ModelicaOpRewritePattern<LteOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(LteOp op, mlir::PatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), castedValues);
      rewriter.replaceOpWithNewOp<LteOp>(op, op.getResult().getType(), castedValues[0], castedValues[1]);
      return mlir::success();
    }
  };

  struct LteOpLowering : public ModelicaOpConversionPattern<LteOp>
  {
    using ModelicaOpConversionPattern<LteOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LteOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (adaptor.getLhs().getType() != adaptor.getRhs().getType()) {
        return rewriter.notifyMatchFailure(op, "Unequal operand types");
      }

      auto loc = op.getLoc();
      mlir::Type operandsType = adaptor.getLhs().getType();

      if (operandsType.isa<mlir::IntegerType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpIPredicate::sle, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);

        if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
          result = rewriter.create<CastOp>(loc, resultType, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return mlir::failure();
    }
  };
}

//===----------------------------------------------------------------------===//
// Logic operations
//===----------------------------------------------------------------------===//

namespace
{
  struct NotOpIntegerLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(NotOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(operandType).isa<mlir::IntegerType>());
    }

    void rewrite(NotOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

      mlir::Value result = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::eq,
          adaptor.getOperand(), zero);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct NotOpFloatLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(NotOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(operandType).isa<mlir::FloatType>());
    }

    void rewrite(NotOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));

      mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::OEQ,
          adaptor.getOperand(), zero);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct NotOpLowering : public ModelicaOpConversionPattern<NotOp>
  {
    using ModelicaOpConversionPattern<NotOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(NotOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operand is compatible
      if (!op.getOperand().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not a Boolean");
      }

      // There is no native negate operation in LLVM IR, so we need to leverage
      // a property of the XOR operation: x XOR true = NOT x
      mlir::Value trueValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      mlir::Value result = rewriter.create<mlir::arith::XOrIOp>(loc, trueValue, adaptor.getOperand());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct NotOpArrayLowering : public ModelicaOpRewritePattern<NotOp>
  {
    using ModelicaOpRewritePattern<NotOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NotOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operand is compatible
      if (!op.getOperand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array");
      }

      if (auto operandArrayType = op.getOperand().getType().cast<ArrayType>(); !operandArrayType.getElementType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Operand is not an array of booleans");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getOperand());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array element
      iterateArray(
          rewriter, loc, op.getOperand(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value value = nestedBuilder.create<LoadOp>(location, op.getOperand(), indices);
            mlir::Value negated = nestedBuilder.create<NotOp>(location, resultArrayType.getElementType(), value);
            nestedBuilder.create<StoreOp>(location, negated, result, indices);
          });

      return mlir::success();
    }
  };

  template<typename Op>
  struct BinaryLogicOpArrayLowering : public ModelicaOpRewritePattern<Op>
  {
    using ModelicaOpRewritePattern<Op>::ModelicaOpRewritePattern;

    virtual mlir::Value getLhs(Op op) const = 0;
    virtual mlir::Value getRhs(Op op) const = 0;

    virtual mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const = 0;

    mlir::LogicalResult matchAndRewrite(Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!getLhs(op).getType().template isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array");
      }

      auto lhsArrayType = getLhs(op).getType().template cast<ArrayType>();

      if (!lhsArrayType.getElementType().template isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not an array of booleans");
      }

      if (!getRhs(op).getType().template isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array");
      }

      auto rhsArrayType = getRhs(op).getType().template cast<ArrayType>();

      if (!rhsArrayType.getElementType().template isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not an array of booleans");
      }

      assert(lhsArrayType.getRank() == rhsArrayType.getRank());

      if (this->options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == ArrayType::kDynamicSize || rhsShape[i] == ArrayType::kDynamicSize) {
            mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, getLhs(op), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, getRhs(op), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().template cast<ArrayType>();
      auto lhsDynamicDimensions = getArrayDynamicDimensions(rewriter, loc, getLhs(op));
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, lhsDynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, result,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(loc, getLhs(op), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(loc, getRhs(op), indices);
            mlir::Value scalarResult = scalarize(nestedBuilder, location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(loc, scalarResult, result, indices);
          });

      return mlir::success();
    }
  };

  struct AndOpIntegersLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AndOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::IntegerType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::IntegerType>());
    }

    void rewrite(AndOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct AndOpFloatsLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AndOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::FloatType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::FloatType>());
    }

    void rewrite(AndOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct AndOpIntegerFloatLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AndOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::IntegerType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::FloatType>());
    }

    void rewrite(AndOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct AndOpFloatIntegerLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AndOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::FloatType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::IntegerType>());
    }

    void rewrite(AndOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
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

    mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const override
    {
      return builder.create<AndOp>(loc, elementType, lhs, rhs);
    }
  };

  struct OrOpIntegersLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(OrOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::IntegerType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::IntegerType>());
    }

    void rewrite(OrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct OrOpFloatsLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(OrOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::FloatType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::FloatType>());
    }

    void rewrite(OrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct OrOpIntegerFloatLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(OrOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::IntegerType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::FloatType>());
    }

    void rewrite(OrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
    }
  };

  struct OrOpFloatIntegerLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(OrOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      return mlir::LogicalResult::success(
          getTypeConverter()->convertType(lhsType).isa<mlir::FloatType>() &&
          getTypeConverter()->convertType(rhsType).isa<mlir::IntegerType>());
    }

    void rewrite(OrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value lhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getLhs().getType()));

      mlir::Value lhs = rewriter.create<mlir::arith::CmpFOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpFPredicate::ONE,
          adaptor.getLhs(), lhsZero);

      mlir::Value rhsZero = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(adaptor.getRhs().getType()));

      mlir::Value rhs = rewriter.create<mlir::arith::CmpIOp>(
          loc, rewriter.getIntegerType(1),
          mlir::arith::CmpIPredicate::ne,
          adaptor.getRhs(), rhsZero);

      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(
          loc, rewriter.getIntegerType(1), lhs, rhs);

      result = getTypeConverter()->materializeSourceConversion(
          rewriter, loc, BooleanType::get(op->getContext()), result);

      if (auto resultType = op.getResult().getType(); result.getType() != resultType) {
        result = rewriter.create<CastOp>(loc, resultType, result);
      }

      rewriter.replaceOp(op, result);
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

    mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const override
    {
      return builder.create<OrOp>(loc, elementType, lhs, rhs);
    }
  };
}

//===----------------------------------------------------------------------===//
// Math operations
//===----------------------------------------------------------------------===//

namespace
{
  struct ConstantOpLowering : public ModelicaOpConversionPattern<ConstantOp>
  {
    public:
    using ModelicaOpConversionPattern<ConstantOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto attribute = convertAttribute(rewriter, op.getResult().getType(), op.getValue());
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, attribute);
      return mlir::success();
    }

    private:
    mlir::Attribute convertAttribute(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Attribute attribute) const
    {
      if (attribute.getType().isa<mlir::IndexType>()) {
        return attribute;
      }

      resultType = getTypeConverter()->convertType(resultType);

      if (auto booleanAttribute = attribute.dyn_cast<BooleanAttr>()) {
        return builder.getBoolAttr(booleanAttribute.getValue());
      }

      if (auto integerAttribute = attribute.dyn_cast<IntegerAttr>()) {
        return builder.getIntegerAttr(resultType, integerAttribute.getValue());
      }

      if (auto realAttribute = attribute.dyn_cast<RealAttr>()) {
        return builder.getFloatAttr(resultType, realAttribute.getValue());
      }

      llvm_unreachable("Unknown attribute type");
      return nullptr;
    }
  };

  struct NegateOpLowering : public ModelicaOpConversionPattern<NegateOp>
  {
    using ModelicaOpConversionPattern<NegateOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(NegateOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Type type = op.getOperand().getType();

      // Check if the operand is compatible
      if (!isNumeric(op.getOperand())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");
      }

      // Compute the result
      if (type.isa<mlir::IndexType, BooleanType, IntegerType>()) {
        mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getZeroAttr(adaptor.getOperand().getType()));
        mlir::Value result = rewriter.create<mlir::arith::SubIOp>(loc, zeroValue, adaptor.getOperand());
        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      if (type.isa<RealType>()) {
        mlir::Value result = rewriter.create<mlir::arith::NegFOp>(loc, adaptor.getOperand());
        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, type, result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown type");
    }
  };

  struct NegateOpArrayLowering : public ModelicaOpRewritePattern<NegateOp>
  {
    using ModelicaOpRewritePattern<NegateOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NegateOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operand is compatible
      if (!op.getOperand().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Value is not an array");
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getOperand());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getOperand(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value source = nestedBuilder.create<LoadOp>(location, op.getOperand(), indices);
            mlir::Value value = nestedBuilder.create<NegateOp>(location, resultArrayType.getElementType(), source);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });

      return mlir::success();
    }
  };

  struct AddOpOpCastPattern : public ModelicaOpRewritePattern<AddOp>
  {
    using ModelicaOpRewritePattern<AddOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
        // Operands must be scalar values
        return mlir::failure();
      }

      if (lhsType != rhsType) {
        return mlir::success();
      }

      if (auto result = op.getResult().getType(); result != lhsType || result != rhsType) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(AddOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> values;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), values);
      assert(values[0].getType() == values[1].getType());

      mlir::Value result = rewriter.create<AddOp>(loc, values[0].getType(), values[0], values[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AddOpIntegerLowering : public ModelicaOpConversionPattern<AddOp>
  {
    using ModelicaOpConversionPattern<AddOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an integer");
      }

      if (!rhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an integer");
      }

      if (!resultType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an integer");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  struct AddOpFloatLowering : public ModelicaOpConversionPattern<AddOp>
  {
    using ModelicaOpConversionPattern<AddOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a float");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a float");
      }

      if (!resultType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not a float");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  struct AddOpArraysLowering : public ModelicaOpRewritePattern<AddOp>
  {
    using ModelicaOpRewritePattern<AddOp>::ModelicaOpRewritePattern;

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
        // Incompatible ranks
        return mlir::failure();
      }

      for (const auto& [lhsDimension, rhsDimension] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDimension != ArrayType::kDynamicSize && rhsDimension != ArrayType::kDynamicSize && lhsDimension != rhsDimension) {
          // Incompatible array dimensions
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    void rewrite(AddOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == ArrayType::kDynamicSize || rhsShape[i] == ArrayType::kDynamicSize) {
            mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.getLhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.getRhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  struct AddEWOpScalarsLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values
      return mlir::LogicalResult::success(isNumeric(lhsType) && isNumeric(rhsType));
    }

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct AddEWOpArraysLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be arrays
      return mlir::LogicalResult::success(lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>());
    }

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct AddEWOpMixedLowering : public ModelicaOpRewritePattern<AddEWOp>
  {
    using ModelicaOpRewritePattern<AddEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AddEWOp op) const override
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

    void rewrite(AddEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      mlir::Value array = op.getLhs().getType().isa<ArrayType>() ? op.getLhs() : op.getRhs();
      mlir::Value scalar = op.getLhs().getType().isa<ArrayType>() ? op.getRhs() : op.getLhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<AddOp>(location, resultArrayType.getElementType(), arrayValue, scalar);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  struct SubOpOpCastPattern : public ModelicaOpRewritePattern<SubOp>
  {
    using ModelicaOpRewritePattern<SubOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
        // Operands must be scalar values
        return mlir::failure();
      }

      if (lhsType != rhsType) {
        return mlir::success();
      }

      if (auto result = op.getResult().getType(); result != lhsType || result != rhsType) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> values;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), values);
      assert(values[0].getType() == values[1].getType());

      mlir::Value result = rewriter.create<SubOp>(loc, values[0].getType(), values[0], values[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SubOpIntegerLowering : public ModelicaOpConversionPattern<SubOp>
  {
    using ModelicaOpConversionPattern<SubOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SubOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an integer");
      }

      if (!rhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an integer");
      }

      if (!resultType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an integer");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  struct SubOpFloatLowering : public ModelicaOpConversionPattern<SubOp>
  {
    using ModelicaOpConversionPattern<SubOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SubOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a float");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a float");
      }

      if (!resultType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not a float");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  struct SubOpArraysLowering : public ModelicaOpRewritePattern<SubOp>
  {
    using ModelicaOpRewritePattern<SubOp>::ModelicaOpRewritePattern;

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
        // Incompatible ranks
        return mlir::failure();
      }

      for (const auto& [lhsDimension, rhsDimension] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDimension != ArrayType::kDynamicSize && rhsDimension != ArrayType::kDynamicSize && lhsDimension != rhsDimension) {
          // Incompatible array dimensions
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    void rewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == ArrayType::kDynamicSize || rhsShape[i] == ArrayType::kDynamicSize) {
            mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.getLhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.getRhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
            mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  struct SubEWOpScalarsLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values
      return mlir::LogicalResult::success(isNumeric(lhsType) && isNumeric(rhsType));
    }

    void rewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct SubEWOpArraysLowering : public ModelicaOpRewritePattern<SubEWOp>
  {
    using ModelicaOpRewritePattern<SubEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SubEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be arrays
      return mlir::LogicalResult::success(lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>());
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

      if (lhsType.isa<ArrayType>() && !rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      if (!lhsType.isa<ArrayType>() && rhsType.isa<ArrayType>()) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(SubEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs().getType().isa<ArrayType>() ? op.getLhs() : op.getRhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            if (op.getLhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), arrayValue, op.getRhs());
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
              mlir::Value value = nestedBuilder.create<SubOp>(location, resultArrayType.getElementType(), op.getLhs(), arrayValue);
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });
    }
  };

  struct MulOpOpCastPattern : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MulOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
        // Operands must be scalar values
        return mlir::failure();
      }

      if (lhsType != rhsType) {
        return mlir::success();
      }

      if (auto result = op.getResult().getType(); result != lhsType || result != rhsType) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> values;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), values);
      assert(values[0].getType() == values[1].getType());

      mlir::Value result = rewriter.create<MulOp>(loc, values[0].getType(), values[0], values[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  /// Product between two scalar integer values.
  struct MulOpIntegerLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an integer");
      }

      if (!rhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an integer");
      }

      if (!resultType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an integer");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  /// Product between two scalar float values.
  struct MulOpFloatLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a float");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a float");
      }

      if (!resultType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not a float");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  /// Product between a scalar and an array.
  struct MulOpScalarProductLowering : public ModelicaOpRewritePattern<MulOp>
  {
    using ModelicaOpRewritePattern<MulOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<ArrayType>() && !op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: none of the operands is an array");
      }

      if (op.getLhs().getType().isa<ArrayType>() && !isNumeric(op.getRhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: right-hand side operand is not a scalar");
      }

      if (op.getRhs().getType().isa<ArrayType>() && !isNumeric(op.getLhs().getType())) {
        return rewriter.notifyMatchFailure(op, "Scalar-array product: left-hand side operand is not a scalar");
      }

      mlir::Value array = op.getLhs().getType().isa<ArrayType>() ? op.getLhs() : op.getRhs();
      mlir::Value scalar = op.getLhs().getType().isa<ArrayType>() ? op.getRhs() : op.getLhs();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, array);
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Multiply each array element by the scalar value
      iterateArray(
          rewriter, loc, array,
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, array, indices);
            mlir::Value value = nestedBuilder.create<MulOp>(location, resultArrayType.getElementType(), scalar, arrayValue);
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
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Cross product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Cross product: left-hand side arrays is not 1-D");
      }

      if (!op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Cross product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Cross product: right-hand side arrays is not 1-D");
      }

      if (lhsArrayType.getShape()[0] != ArrayType::kDynamicSize && rhsArrayType.getShape()[0] != ArrayType::kDynamicSize) {
        if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Cross product: the two arrays have different shape");
        }
      }

      assert(lhsArrayType.getRank() == 1);
      assert(rhsArrayType.getRank() == 1);

      mlir::Value lhs = op.getLhs();
      mlir::Value rhs = op.getRhs();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[0] == ArrayType::kDynamicSize || rhsShape[0] == ArrayType::kDynamicSize) {
          mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, lhs, dimensionIndex);
          mlir::Value rhsDimensionSize =  rewriter.create<DimOp>(loc, rhs, dimensionIndex);
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Compute the result
      mlir::Type resultType = op.getResult().getType();

      mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(getTypeConverter()->convertType(resultType)));

      // Iterate on the two arrays at the same time, and propagate the
      // progressive result to the next loop iteration.
      auto loop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());

        mlir::Value lhsScalarValue = rewriter.create<LoadOp>(loc, lhs, loop.getInductionVar());
        mlir::Value rhsScalarValue = rewriter.create<LoadOp>(loc, rhs, loop.getInductionVar());
        mlir::Value product = rewriter.create<MulOp>(loc, resultType, lhsScalarValue, rhsScalarValue);

        mlir::Value accumulatedValue = loop.getRegionIterArgs()[0];

        accumulatedValue = getTypeConverter()->materializeSourceConversion(
            rewriter, accumulatedValue.getLoc(), resultType, accumulatedValue);

        mlir::Value sum = rewriter.create<AddOp>(loc, resultType, product, accumulatedValue);

        sum = getTypeConverter()->materializeTargetConversion(
            rewriter, sum.getLoc(), getTypeConverter()->convertType(sum.getType()), sum);

        rewriter.create<mlir::scf::YieldOp>(loc, sum);
      }

      rewriter.replaceOp(op, loop.getResult(0));

      //rewriter.replaceOp(op, init);
      return mlir::success();
    }
  };

  /// Product of a vector (1-D array) and a matrix (2-D array).
  ///
  /// [ x1 ]  *  [ y11, y12 ]  =  [ (x1 * y11 + x2 * y21 + x3 * y31), (x1 * y12 + x2 * y22 + x3 * y32) ]
  /// [ x2 ]		 [ y21, y22 ]
  /// [ x3 ]		 [ y31, y32 ]
  struct MulOpVectorMatrixLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: left-hand size array is not 1-D");
      }

      if (!op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Vector-matrix product: right-hand side matrix is not 2-D");
      }

      if (lhsArrayType.getShape()[0] != ArrayType::kDynamicSize && rhsArrayType.getShape()[0] != ArrayType::kDynamicSize) {
        if (lhsArrayType.getShape()[0] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Vector-matrix product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 1);
      assert(rhsArrayType.getRank() == 2);

      mlir::Value lhs = op.getLhs();
      mlir::Value rhs = op.getRhs();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[0] == ArrayType::kDynamicSize || rhsShape[0] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, lhs, zero);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, rhs, zero);
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();
      assert(shape.size() == 1);

      llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

      if (shape[0] == ArrayType::kDynamicSize) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the columns
      mlir::Value columnsLowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1)));
      mlir::Value columnsStep = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto outerLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      // Product between the vector and the current column
      mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(getTypeConverter()->convertType(resultArrayType.getElementType())));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhsScalarValue = rewriter.create<LoadOp>(loc, lhs, innerLoop.getInductionVar());
      mlir::Value rhsScalarValue = rewriter.create<LoadOp>(loc, rhs, mlir::ValueRange({ innerLoop.getInductionVar(), outerLoop.getInductionVar() }));
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhsScalarValue, rhsScalarValue);

      mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

      accumulatedValue = getTypeConverter()->materializeSourceConversion(
          rewriter, accumulatedValue.getLoc(), resultArrayType.getElementType(), accumulatedValue);

      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, accumulatedValue);

      sum = getTypeConverter()->materializeTargetConversion(
          rewriter, sum.getLoc(), getTypeConverter()->convertType(sum.getType()), sum);

      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);

      productResult = getTypeConverter()->materializeSourceConversion(
          rewriter, productResult.getLoc(), resultArrayType.getElementType(), productResult);

      rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVar());

      rewriter.setInsertionPointAfter(outerLoop);
      return mlir::success();
    }
  };

  /// Product of a matrix (2-D array) and a vector (1-D array).
  ///
  /// [ x11, x12 ] * [ y1, y2 ] = [ x11 * y1 + x12 * y2 ]
  /// [ x21, x22 ]							  [ x21 * y1 + x22 * y2 ]
  /// [ x31, x32 ]								[ x31 * y1 + x22 * y2 ]
  struct MulOpMatrixVectorLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: left-hand size array is not 2-D");
      }

      if (!op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 1) {
        return rewriter.notifyMatchFailure(op, "Matrix-vector product: right-hand side matrix is not 1-D");
      }

      if (lhsArrayType.getShape()[1] != ArrayType::kDynamicSize && rhsArrayType.getShape()[0] != ArrayType::kDynamicSize) {
        if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 2);
      assert(rhsArrayType.getRank() == 1);

      mlir::Value lhs = op.getLhs();
      mlir::Value rhs = op.getRhs();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[1] == ArrayType::kDynamicSize || rhsShape[0] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, lhs, one);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, rhs, zero);
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();

      llvm::SmallVector<mlir::Value, 1> dynamicDimensions;

      if (shape[0] == ArrayType::kDynamicSize) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the rows
      mlir::Value rowsLowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value rowsStep = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto outerLoop = rewriter.create<mlir::scf::ParallelOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
      rewriter.setInsertionPointToStart(outerLoop.getBody());

      // Product between the current row and the vector
      mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(getTypeConverter()->convertType(resultArrayType.getElementType())));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhsScalarValue = rewriter.create<LoadOp>(loc, lhs, mlir::ValueRange({ outerLoop.getInductionVars()[0], innerLoop.getInductionVar() }));
      mlir::Value rhsScalarValue = rewriter.create<LoadOp>(loc, rhs, innerLoop.getInductionVar());
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhsScalarValue, rhsScalarValue);

      mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

      accumulatedValue = getTypeConverter()->materializeSourceConversion(
          rewriter, accumulatedValue.getLoc(), resultArrayType.getElementType(), accumulatedValue);

      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, accumulatedValue);

      sum = getTypeConverter()->materializeTargetConversion(
          rewriter, sum.getLoc(), getTypeConverter()->convertType(sum.getType()), sum);

      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);

      productResult = getTypeConverter()->materializeSourceConversion(
          rewriter, productResult.getLoc(), resultArrayType.getElementType(), productResult);

      rewriter.create<StoreOp>(loc, productResult, result, outerLoop.getInductionVars()[0]);
      rewriter.setInsertionPointAfter(outerLoop);

      return mlir::success();
    }
  };

  /// Product of two matrices (2-D arrays).
  ///
  /// [ x11, x12, x13 ] * [ y11, y12 ]  =  [ x11 * y11 + x12 * y21 + x13 * y31, x11 * y12 + x12 * y22 + x13 * y32 ]
  /// [ x21, x22, x23 ]   [ y21, y22 ]		 [ x21 * y11 + x22 * y21 + x23 * y31, x21 * y12 + x22 * y22 + x23 * y32 ]
  /// [ x31, x32, x33 ]	  [ y31, y32 ]		 [ x31 * y11 + x32 * y21 + x33 * y31, x31 * y12 + x32 * y22 + x33 * y32 ]
  /// [ x41, x42, x43 ]
  struct MulOpMatrixLowering : public ModelicaOpConversionPattern<MulOp>
  {
    using ModelicaOpConversionPattern<MulOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix product: left-hand side value is not an array");
      }

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();

      if (lhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix product: left-hand size array is not 2-D");
      }

      if (!op.getRhs().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side value is not an array");
      }

      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (rhsArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Matrix product: right-hand side matrix is not 2-D");
      }

      if (lhsArrayType.getShape()[1] != ArrayType::kDynamicSize && rhsArrayType.getShape()[0] != ArrayType::kDynamicSize) {
        if (lhsArrayType.getShape()[1] != rhsArrayType.getShape()[0]) {
          return rewriter.notifyMatchFailure(op, "Matrix-vector product: incompatible shapes");
        }
      }

      assert(lhsArrayType.getRank() == 2);
      assert(rhsArrayType.getRank() == 2);

      mlir::Value lhs = op.getLhs();
      mlir::Value rhs = op.getRhs();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        if (lhsShape[1] == ArrayType::kDynamicSize || rhsShape[0] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, lhs, one);
          mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, rhs, zero);
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto shape = resultArrayType.getShape();

      llvm::SmallVector<mlir::Value, 2> dynamicDimensions;

      if (shape[0] == ArrayType::kDynamicSize) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0))));
      }

      if (shape[1] == ArrayType::kDynamicSize) {
        dynamicDimensions.push_back(rewriter.create<DimOp>(
            loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1))));
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Iterate on the rows
      mlir::Value rowsLowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value rowsUpperBound = rewriter.create<DimOp>(loc, lhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value rowsStep = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto rowsLoop = rewriter.create<mlir::scf::ForOp>(loc, rowsLowerBound, rowsUpperBound, rowsStep);
      rewriter.setInsertionPointToStart(rowsLoop.getBody());

      // Iterate on the columns
      mlir::Value columnsLowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value columnsUpperBound = rewriter.create<DimOp>(loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1)));
      mlir::Value columnsStep = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto columnsLoop = rewriter.create<mlir::scf::ForOp>(loc, columnsLowerBound, columnsUpperBound, columnsStep);
      rewriter.setInsertionPointToStart(columnsLoop.getBody());

      // Product between the current row and the current column
      mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
      mlir::Value upperBound = rewriter.create<DimOp>(loc, rhs, rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      mlir::Value init = rewriter.create<mlir::arith::ConstantOp>(
          loc, rewriter.getZeroAttr(getTypeConverter()->convertType(resultArrayType.getElementType())));

      auto innerLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step, init);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      mlir::Value lhsScalarValue = rewriter.create<LoadOp>(loc, lhs, mlir::ValueRange({ rowsLoop.getInductionVar(), innerLoop.getInductionVar() }));
      mlir::Value rhsScalarValue = rewriter.create<LoadOp>(loc, rhs, mlir::ValueRange({ innerLoop.getInductionVar(), columnsLoop.getInductionVar() }));
      mlir::Value product = rewriter.create<MulOp>(loc, resultArrayType.getElementType(), lhsScalarValue, rhsScalarValue);

      mlir::Value accumulatedValue = innerLoop.getRegionIterArgs()[0];

      accumulatedValue = getTypeConverter()->materializeSourceConversion(
          rewriter, accumulatedValue.getLoc(), resultArrayType.getElementType(), accumulatedValue);

      mlir::Value sum = rewriter.create<AddOp>(loc, resultArrayType.getElementType(), product, accumulatedValue);

      sum = getTypeConverter()->materializeTargetConversion(
          rewriter, sum.getLoc(), getTypeConverter()->convertType(sum.getType()), sum);

      rewriter.create<mlir::scf::YieldOp>(loc, sum);

      // Store the product in the result array
      rewriter.setInsertionPointAfter(innerLoop);
      mlir::Value productResult = innerLoop.getResult(0);

      productResult = getTypeConverter()->materializeSourceConversion(
          rewriter, productResult.getLoc(), resultArrayType.getElementType(), productResult);

      rewriter.create<StoreOp>(loc, productResult, result, mlir::ValueRange({ rowsLoop.getInductionVar(), columnsLoop.getInductionVar() }));
      rewriter.setInsertionPointAfter(rowsLoop);

      return mlir::success();
    }
  };

  /// Element-wise product of two scalar values.
  struct MulEWOpScalarsLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MulEWOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      // Operands must be scalar values
      return mlir::LogicalResult::success(isNumeric(lhsType) && isNumeric(rhsType));
    }

    void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  /// Element-wise product of two arrays.
  struct MulEWOpArraysLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

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
        // Incompatible ranks
        return mlir::failure();
      }

      for (const auto& [lhsDimension, rhsDimension] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDimension != ArrayType::kDynamicSize && rhsDimension != ArrayType::kDynamicSize && lhsDimension != rhsDimension) {
          // Incompatible array dimensions
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == ArrayType::kDynamicSize || rhsShape[i] == ArrayType::kDynamicSize) {
            mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.getLhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.getRhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
            mlir::Value value = nestedBuilder.create<MulOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  /// Element-wise product between a scalar value and an array.
  struct MulEWOpMixedLowering : public ModelicaOpRewritePattern<MulEWOp>
  {
    using ModelicaOpRewritePattern<MulEWOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MulEWOp op) const override
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

    void rewrite(MulEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  struct DivOpOpCastPattern : public ModelicaOpRewritePattern<DivOp>
  {
    using ModelicaOpRewritePattern<DivOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivOp op) const override
    {
      mlir::Type lhsType = op.getLhs().getType();
      mlir::Type rhsType = op.getRhs().getType();

      if (!isNumeric(lhsType) || !isNumeric(rhsType)) {
        // Operands must be scalar values
        return mlir::failure();
      }

      if (lhsType != rhsType) {
        return mlir::success();
      }

      if (auto result = op.getResult().getType(); result != lhsType || result != rhsType) {
        return mlir::success();
      }

      return mlir::failure();
    }

    void rewrite(DivOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> values;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getLhs(), op.getRhs() }), values);
      assert(values[0].getType() == values[1].getType());

      mlir::Value result = rewriter.create<DivOp>(loc, values[0].getType(), values[0], values[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  /// Division between two scalar integer values.
  struct DivOpIntegerLowering : public ModelicaOpConversionPattern<DivOp>
  {
    using ModelicaOpConversionPattern<DivOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DivOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not an integer");
      }

      if (!rhsType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not an integer");
      }

      if (!resultType.isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an integer");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtSIOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncIOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(op, resultType, lhs, rhs);
      return mlir::success();
    }
  };

  /// Division between two scalar float values.
  struct DivOpFloatLowering : public ModelicaOpConversionPattern<DivOp>
  {
    using ModelicaOpConversionPattern<DivOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DivOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Type lhsType = adaptor.getLhs().getType();
      mlir::Type rhsType = adaptor.getRhs().getType();
      mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

      if (!lhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a float");
      }

      if (!rhsType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side value is not a float");
      }

      if (!resultType.isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not a float");
      }

      mlir::Value lhs = adaptor.getLhs();
      mlir::Value rhs = adaptor.getRhs();

      if (lhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, lhs);
      } else if (lhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        lhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, lhs);
      }

      if (rhsType.getIntOrFloatBitWidth() < resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::ExtFOp>(loc, resultType, rhs);
      } else if (rhsType.getIntOrFloatBitWidth() > resultType.getIntOrFloatBitWidth()) {
        rhs = rewriter.create<mlir::arith::TruncFOp>(loc, resultType, rhs);
      }

      rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(op, resultType, lhs, rhs);
      return mlir::success();
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

      return mlir::LogicalResult::success(lhsType.isa<ArrayType>() && !rhsType.isa<ArrayType>());
    }

    void rewrite(DivOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs().getType().isa<ArrayType>() ? op.getLhs() : op.getRhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
            mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), arrayValue, op.getRhs());
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
      return mlir::LogicalResult::success(isNumeric(lhsType) && isNumeric(rhsType));
    }

    void rewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<DivOp>(op, op.getResult().getType(), op.getLhs(), op.getRhs());
    }
  };

  /// Element-wise division between two arrays.
  struct DivEWOpArraysLowering : public ModelicaOpRewritePattern<DivEWOp>
  {
    using ModelicaOpRewritePattern<DivEWOp>::ModelicaOpRewritePattern;

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
        // Incompatible ranks
        return mlir::failure();
      }

      for (const auto& [lhsDimension, rhsDimension] : llvm::zip(lhsArrayType.getShape(), rhsArrayType.getShape())) {
        if (lhsDimension != ArrayType::kDynamicSize && rhsDimension != ArrayType::kDynamicSize && lhsDimension != rhsDimension) {
          // Incompatible array dimensions
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    void rewrite(DivEWOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto lhsArrayType = op.getLhs().getType().cast<ArrayType>();
      auto rhsArrayType = op.getRhs().getType().cast<ArrayType>();

      if (options.assertions) {
        // Check if the dimensions are compatible
        auto lhsShape = lhsArrayType.getShape();
        auto rhsShape = rhsArrayType.getShape();

        assert(lhsArrayType.getRank() == rhsArrayType.getRank());

        for (unsigned int i = 0; i < lhsArrayType.getRank(); ++i) {
          if (lhsShape[i] == ArrayType::kDynamicSize || rhsShape[i] == ArrayType::kDynamicSize) {
            mlir::Value dimensionIndex = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
            mlir::Value lhsDimensionSize = rewriter.create<DimOp>(loc, op.getLhs(), dimensionIndex);
            mlir::Value rhsDimensionSize = rewriter.create<DimOp>(loc, op.getRhs(), dimensionIndex);
            mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
            rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Incompatible dimensions"));
          }
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            mlir::Value lhs = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
            mlir::Value rhs = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
            mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), lhs, rhs);
            nestedBuilder.create<StoreOp>(location, value, result, indices);
          });
    }
  };

  /// Element-wise division between a scalar value and an array (and vice versa).
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
      auto loc = op->getLoc();

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getLhs().getType().isa<ArrayType>() ? op.getLhs() : op.getRhs());
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, dynamicDimensions);

      // Apply the operation on each array position
      iterateArray(
          rewriter, loc, op.getLhs(),
          [&](mlir::OpBuilder& nestedBuilder, mlir::Location location, mlir::ValueRange indices) {
            if (op.getLhs().getType().isa<ArrayType>()) {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.getLhs(), indices);
              mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), arrayValue, op.getRhs());
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            } else {
              mlir::Value arrayValue = nestedBuilder.create<LoadOp>(location, op.getRhs(), indices);
              mlir::Value value = nestedBuilder.create<DivOp>(location, resultArrayType.getElementType(), op.getLhs(), arrayValue);
              nestedBuilder.create<StoreOp>(location, value, result, indices);
            }
          });
    }
  };

  struct PowOpMatrixLowering : public ModelicaOpRewritePattern<PowOp>
  {
    using ModelicaOpRewritePattern<PowOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(PowOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.getBase().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Base is not an array");
      }

      auto baseArrayType = op.getBase().getType().cast<ArrayType>();

      if (baseArrayType.getRank() != 2) {
        return rewriter.notifyMatchFailure(op, "Base array is not 2-D");
      }

      if (baseArrayType.getShape()[0] != ArrayType::kDynamicSize && baseArrayType.getShape()[1] != ArrayType::kDynamicSize) {
        if (baseArrayType.getShape()[0] != baseArrayType.getShape()[1]) {
          return rewriter.notifyMatchFailure(op, "Base is not a square matrix");
        }
      }

      assert(baseArrayType.getRank() == 2);

      if (options.assertions) {
        // Check if the matrix is a square one
        auto shape = baseArrayType.getShape();

        if (shape[0] == ArrayType::kDynamicSize || shape[1] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value firstDimensionSize = rewriter.create<DimOp>(loc, op.getBase(), one);
          mlir::Value secondDimensionSize = rewriter.create<DimOp>(loc, op.getBase(), zero);
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, firstDimensionSize, secondDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
        }
      }

      // Allocate the result array
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      auto dynamicDimensions = getArrayDynamicDimensions(rewriter, loc, op.getBase());
      mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      mlir::Value size = rewriter.create<DimOp>(loc, op.getBase(), one);
      mlir::Value result = rewriter.replaceOpWithNewOp<IdentityOp>(op, resultArrayType, size);

      // Compute the result
      mlir::Value exponent = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.getExponent());

      exponent = rewriter.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(), exponent,
          rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1)));

      mlir::Value lowerBound = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto forLoop = rewriter.create<mlir::scf::ForOp>(loc, lowerBound, exponent, step);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forLoop.getBody());
        mlir::Value next = rewriter.create<MulOp>(loc, result.getType(), result, op.getBase());
        rewriter.create<AssignmentOp>(loc, result, next);
      }

      return mlir::success();
    }
  };
}

//===----------------------------------------------------------------------===//
// Various operations
//===----------------------------------------------------------------------===//

namespace
{
  struct SelectOpCastPattern : public ModelicaOpRewritePattern<SelectOp>
  {
    using ModelicaOpRewritePattern<SelectOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SelectOp op) const override
    {
      mlir::Type conditionType = op.getCondition().getType();
      mlir::TypeRange trueValuesTypes = op.getTrueValues().getTypes();
      mlir::TypeRange falseValuesTypes = op.getFalseValues().getTypes();
      auto resultTypes = op.getResultTypes();

      return mlir::LogicalResult::success(
          !conditionType.isa<BooleanType>() ||
          !llvm::all_of(llvm::zip(trueValuesTypes, resultTypes), [](const auto& pair) {
            return std::get<0>(pair) == std::get<1>(pair);
          }) ||
          !llvm::all_of(llvm::zip(falseValuesTypes, resultTypes), [](const auto& pair) {
            return std::get<0>(pair) == std::get<1>(pair);
          }));
    }

    void rewrite(SelectOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value condition = op.getCondition();

      if (!condition.getType().isa<BooleanType>()) {
        condition = rewriter.create<CastOp>(loc, BooleanType::get(op.getContext()), condition);
      }

      llvm::SmallVector<mlir::Value, 1> trueValues;
      llvm::SmallVector<mlir::Value, 1> falseValues;

      for (const auto& [value, resultType] : llvm::zip(op.getTrueValues(), op.getResultTypes())) {
        if (value.getType() != resultType) {
          trueValues.push_back(rewriter.create<CastOp>(loc, resultType, value));
        } else {
          trueValues.push_back(value);
        }
      }

      for (const auto& [value, resultType] : llvm::zip(op.getFalseValues(), op.getResultTypes())) {
        if (value.getType() != resultType) {
          falseValues.push_back(rewriter.create<CastOp>(loc, resultType, value));
        } else {
          falseValues.push_back(value);
        }
      }

      rewriter.replaceOpWithNewOp<SelectOp>(op, op.getResultTypes(), condition, trueValues, falseValues);
    }
  };

  struct SelectOpLowering : public ModelicaOpConversionPattern<SelectOp>
  {
    using ModelicaOpConversionPattern<SelectOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SelectOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 1> results;

      for (const auto& [trueValue, falseValue] : llvm::zip(adaptor.getTrueValues(), adaptor.getFalseValues())) {
        results.push_back(rewriter.create<mlir::arith::SelectOp>(loc, adaptor.getCondition(), trueValue, falseValue));
      }

      rewriter.replaceOp(op, results);
      return mlir::success();
    }
  };
}

static void populateModelicaToArithPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::modelica::TypeConverter& typeConverter,
    ModelicaToArithOptions options)
{
  // Comparison operations
  patterns.insert<
      EqOpCastPattern,
      NotEqOpCastPattern,
      GtOpCastPattern,
      GteOpCastPattern,
      LtOpCastPattern,
      LteOpCastPattern>(context, options);

  patterns.insert<
      EqOpLowering,
      NotEqOpLowering,
      GtOpLowering,
      GteOpLowering,
      LtOpLowering,
      LteOpLowering>(context, typeConverter, options);

  // Logic operations
  patterns.insert<
      NotOpArrayLowering,
      AndOpArrayLowering,
      OrOpArrayLowering>(context, options);

  patterns.insert<
      NotOpIntegerLowering,
      NotOpFloatLowering,
      AndOpIntegersLowering,
      AndOpFloatsLowering,
      AndOpIntegerFloatLowering,
      AndOpFloatIntegerLowering,
      OrOpIntegersLowering,
      OrOpFloatsLowering,
      OrOpIntegerFloatLowering,
      OrOpFloatIntegerLowering>(context, typeConverter, options);

  // Math operations
  patterns.insert<
      NegateOpArrayLowering,
      AddOpOpCastPattern,
      AddOpArraysLowering,
      AddEWOpScalarsLowering,
      AddEWOpArraysLowering,
      AddEWOpMixedLowering,
      SubOpOpCastPattern,
      SubOpArraysLowering,
      SubEWOpScalarsLowering,
      SubEWOpArraysLowering,
      SubEWOpMixedLowering,
      MulOpOpCastPattern,
      MulOpScalarProductLowering,
      MulEWOpScalarsLowering,
      MulEWOpArraysLowering,
      MulEWOpMixedLowering,
      DivOpOpCastPattern,
      DivOpMixedLowering,
      DivEWOpScalarsLowering,
      DivEWOpArraysLowering,
      DivEWOpMixedLowering,
      PowOpMatrixLowering>(context, options);

  patterns.insert<
      ConstantOpLowering,
      NegateOpLowering,
      AddOpIntegerLowering,
      AddOpFloatLowering,
      SubOpIntegerLowering,
      SubOpFloatLowering,
      MulOpIntegerLowering,
      MulOpFloatLowering,
      MulOpCrossProductLowering,
      MulOpVectorMatrixLowering,
      MulOpMatrixVectorLowering,
      MulOpMatrixLowering,
      DivOpIntegerLowering,
      DivOpFloatLowering>(context, typeConverter, options);

  // Various operations
  patterns.insert<
      SelectOpCastPattern>(context, options);

  patterns.insert<
      SelectOpLowering>(context, typeConverter, options);
}

namespace
{
  class ModelicaToArithConversionPass : public ModelicaToArithBase<ModelicaToArithConversionPass>
  {
    public:
      ModelicaToArithConversionPass(ModelicaToArithOptions options)
          : options(std::move(options))
      {
      }

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

        target.addIllegalOp<
            ConstantOp,
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

        target.addIllegalOp<
            SelectOp>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        llvmLoweringOptions.dataLayout = options.dataLayout;
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions, options.bitWidth);

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToArithPatterns(patterns, &getContext(), typeConverter, options);

        return applyPartialConversion(module, target, std::move(patterns));
      }

    private:
      ModelicaToArithOptions options;
  };
}

namespace marco::codegen
{
  const ModelicaToArithOptions& ModelicaToArithOptions::getDefaultOptions()
  {
    static ModelicaToArithOptions options;
    return options;
  }

  std::unique_ptr<mlir::Pass> createModelicaToArithPass(ModelicaToArithOptions options)
  {
    return std::make_unique<ModelicaToArithConversionPass>(options);
  }
}
