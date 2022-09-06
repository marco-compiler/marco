#include "marco/Codegen/Conversion/ModelicaToVector/ModelicaToVector.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOVECTORCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static mlir::Value readVectorFromMemRef(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value memRef)
{
  auto memRefType = memRef.getType().cast<mlir::MemRefType>();
  auto vectorType = mlir::VectorType::get(memRefType.getShape(), memRefType.getElementType());

  mlir::Value zeroIndex = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::Value, 3> indices(memRefType.getRank(), zeroIndex);

  return builder.create<mlir::vector::TransferReadOp>(loc, vectorType, memRef, indices);
}

static void writeVectortoMemRef(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value vector, mlir::Value memRef)
{
  auto memRefType = memRef.getType().cast<mlir::MemRefType>();
  mlir::Value zeroIndex = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  llvm::SmallVector<mlir::Value, 3> indices(memRefType.getRank(), zeroIndex);

  builder.create<mlir::vector::TransferWriteOp>(loc, vector, memRef, indices);
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

    protected:
      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
      }
  };
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace
{
  template<typename... ElementType>
  struct AddOpArraysLowering : public ModelicaOpConversionPattern<AddOp>
  {
    public:
      using ModelicaOpConversionPattern::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto loc = op.getLoc();

        auto resultConvertedType = getTypeConverter()->convertType(op.getResult().getType());
        mlir::Type elementType = resultConvertedType.template cast<mlir::MemRefType>().getElementType();

        if (!mlir::isa<ElementType...>(elementType)) {
          return mlir::failure();
        }

        mlir::Value lhs = readVectorFromMemRef(rewriter, loc, adaptor.getLhs());
        mlir::Value rhs = readVectorFromMemRef(rewriter, loc, adaptor.getRhs());

        mlir::Value resultVector = createVectorOp(rewriter, loc, lhs, rhs);

        mlir::Value resultArray = rewriter.replaceOpWithNewOp<AllocOp>(op, op.getResult().getType().cast<ArrayType>(), llvm::None);
        mlir::Value resultMemRef = materializeTargetConversion(rewriter, resultArray);

        writeVectortoMemRef(rewriter, loc, resultVector, resultMemRef);
        return mlir::success();
      }

    protected:
      virtual mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
  };

  struct AddOpIntegerLikeArraysLowering : public AddOpArraysLowering<mlir::IndexType, mlir::IntegerType>
  {
    public:
      using AddOpArraysLowering::AddOpArraysLowering;

      mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
      {
        return builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
      }
  };

  struct AddOpFloatArraysLowering : public AddOpArraysLowering<mlir::FloatType>
  {
    public:
      using AddOpArraysLowering::AddOpArraysLowering;

      mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
      {
        return builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
      }
  };
}

//===----------------------------------------------------------------------===//
// ProductOp
//===----------------------------------------------------------------------===//

namespace
{
  struct ProductOpCastPattern : public ModelicaOpRewritePattern<ProductOp>
  {
    using ModelicaOpRewritePattern::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ProductOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(ProductOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto resultType = op.getArray().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<ProductOp>(loc, resultType, op.getArray());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  template<typename... ElementType>
  struct ProductOpLowering : public ModelicaOpConversionPattern<ProductOp>
  {
    public:
      using ModelicaOpConversionPattern::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(ProductOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Type elementType = adaptor.getArray().getType().template cast<mlir::MemRefType>().getElementType();
        mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

        if (elementType != resultType || !mlir::isa<ElementType...>(resultType)) {
          return mlir::failure();
        }

        auto loc = op.getLoc();
        mlir::Value array = readVectorFromMemRef(rewriter, loc, adaptor.getArray());
        mlir::Value acc = createAccumulator(rewriter, loc, resultType);

        int64_t rank = array.getType().template cast<mlir::VectorType>().getRank();

        if (rank == 1) {
          rewriter.template replaceOpWithNewOp<mlir::vector::ReductionOp>(
              op, mlir::vector::CombiningKind::MUL, array, acc);
        } else {
          llvm::SmallVector<bool, 2> reductionMask(rank, true);

          rewriter.template replaceOpWithNewOp<mlir::vector::MultiDimReductionOp>(
              op, array, acc, reductionMask, mlir::vector::CombiningKind::MUL);
        }

        return mlir::success();
      }

    protected:
      virtual mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const = 0;
  };

  struct ProductOpIntegerLikeLowering : public ProductOpLowering<mlir::IndexType, mlir::IntegerType>
  {
    public:
      using ProductOpLowering::ProductOpLowering;

    protected:
      mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const override
      {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(type, 1));
      }
  };

  struct ProductOpFloatLowering : public ProductOpLowering<mlir::FloatType>
  {
    public:
      using ProductOpLowering::ProductOpLowering;

    protected:
      mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const override
      {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(type, 1));
      }
  };
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

namespace
{
  template<typename... ElementType>
  struct SubOpArraysLowering : public ModelicaOpConversionPattern<SubOp>
  {
    public:
      using ModelicaOpConversionPattern::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(SubOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto loc = op.getLoc();

        auto resultConvertedType = getTypeConverter()->convertType(op.getResult().getType());
        mlir::Type elementType = resultConvertedType.template cast<mlir::MemRefType>().getElementType();

        if (!mlir::isa<ElementType...>(elementType)) {
          return mlir::failure();
        }

        mlir::Value lhs = readVectorFromMemRef(rewriter, loc, adaptor.getLhs());
        mlir::Value rhs = readVectorFromMemRef(rewriter, loc, adaptor.getRhs());

        mlir::Value resultVector = createVectorOp(rewriter, loc, lhs, rhs);

        mlir::Value resultArray = rewriter.replaceOpWithNewOp<AllocOp>(op, op.getResult().getType().cast<ArrayType>(), llvm::None);
        mlir::Value resultMemRef = materializeTargetConversion(rewriter, resultArray);

        writeVectortoMemRef(rewriter, loc, resultVector, resultMemRef);
        return mlir::success();
      }

    protected:
      virtual mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const = 0;
  };

  struct SubOpIntegerLikeArraysLowering : public SubOpArraysLowering<mlir::IndexType, mlir::IntegerType>
  {
    public:
      using SubOpArraysLowering::SubOpArraysLowering;

      mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
      {
        return builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      }
  };

  struct SubOpFloatArraysLowering : public SubOpArraysLowering<mlir::FloatType>
  {
    public:
      using SubOpArraysLowering::SubOpArraysLowering;

      mlir::Value createVectorOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) const override
      {
        return builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
      }
  };
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

namespace
{
  struct SumOpCastPattern : public ModelicaOpRewritePattern<SumOp>
  {
    using ModelicaOpRewritePattern::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SumOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(SumOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto resultType = op.getArray().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<SumOp>(loc, resultType, op.getArray());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  template<typename... ElementType>
  struct SumOpLowering : public ModelicaOpConversionPattern<SumOp>
  {
    public:
      using ModelicaOpConversionPattern::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(SumOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Type elementType = adaptor.getArray().getType().template cast<mlir::MemRefType>().getElementType();
        mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());

        if (elementType != resultType || !mlir::isa<ElementType...>(resultType)) {
          return mlir::failure();
        }

        auto loc = op.getLoc();
        mlir::Value array = readVectorFromMemRef(rewriter, loc, adaptor.getArray());
        mlir::Value acc = createAccumulator(rewriter, loc, resultType);

        int64_t rank = array.getType().template cast<mlir::VectorType>().getRank();

        if (rank == 1) {
          rewriter.template replaceOpWithNewOp<mlir::vector::ReductionOp>(
              op, mlir::vector::CombiningKind::ADD, array, acc);
        } else {
          llvm::SmallVector<bool, 2> reductionMask(rank, true);

          rewriter.template replaceOpWithNewOp<mlir::vector::MultiDimReductionOp>(
              op, array, acc, reductionMask, mlir::vector::CombiningKind::ADD);
        }

        return mlir::success();
      }

    protected:
      virtual mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const = 0;
  };

  struct SumOpIntegerLikeLowering : public SumOpLowering<mlir::IndexType, mlir::IntegerType>
  {
    public:
      using SumOpLowering::SumOpLowering;

    protected:
      mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const override
      {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(type, 0));
      }
  };

  struct SumOpFloatLowering : public SumOpLowering<mlir::FloatType>
  {
    public:
      using SumOpLowering::SumOpLowering;

    protected:
      mlir::Value createAccumulator(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type type) const override
      {
        return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(type, 0));
      }
  };
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace
{
  struct TransposeOpLowering : public ModelicaOpConversionPattern<TransposeOp>
  {
    public:
      using ModelicaOpConversionPattern::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto loc = op.getLoc();

        mlir::Value matrix = readVectorFromMemRef(rewriter, loc, adaptor.getMatrix());

        llvm::SmallVector<int64_t, 2> permutation({1, 0});
        mlir::Value resultVector = rewriter.create<mlir::vector::TransposeOp>(loc, matrix, permutation);

        mlir::Value resultArray = rewriter.replaceOpWithNewOp<AllocOp>(op, op.getResult().getType().cast<ArrayType>(), llvm::None);
        mlir::Value resultMemRef = materializeTargetConversion(rewriter, resultArray);

        writeVectortoMemRef(rewriter, loc, resultVector, resultMemRef);
        return mlir::success();
      }
  };
}

static void populateModelicaToVectorPatterns(
    mlir::RewritePatternSet& patterns,
    mlir::MLIRContext* context,
    mlir::TypeConverter& typeConverter)
{
  patterns.insert<
      AddOpIntegerLikeArraysLowering,
      AddOpFloatArraysLowering>(typeConverter, context);

  patterns.insert<
      ProductOpCastPattern>(context);

  patterns.insert<
      ProductOpIntegerLikeLowering,
      ProductOpFloatLowering>(typeConverter, context);

  patterns.insert<
      SubOpIntegerLikeArraysLowering,
      SubOpFloatArraysLowering>(typeConverter, context);

  patterns.insert<
      SumOpCastPattern>(context);

  patterns.insert<
      SumOpIntegerLikeLowering,
      SumOpFloatLowering>(typeConverter, context);

  patterns.insert<
      TransposeOpLowering>(typeConverter, context);
}

namespace
{
  class ModelicaToVectorConversionPass : public mlir::impl::ModelicaToVectorConversionPassBase<ModelicaToVectorConversionPass>
  {
    public:
      using ModelicaToVectorConversionPassBase::ModelicaToVectorConversionPassBase;

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

        target.addLegalDialect<mlir::BuiltinDialect>();
        target.addLegalDialect<ModelicaDialect>();
        target.addLegalDialect<mlir::arith::ArithmeticDialect>();
        target.addLegalDialect<mlir::vector::VectorDialect>();

        target.addDynamicallyLegalOp<AddOp>([](AddOp op) {
          mlir::Type lhsType = op.getLhs().getType();
          mlir::Type rhsType = op.getRhs().getType();

          if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
            return true;
          }

          auto lhsArrayType = lhsType.cast<ArrayType>();
          auto rhsArrayType = rhsType.cast<ArrayType>();
          auto resultArrayType = op.getResult().getType().cast<ArrayType>();

          if (!lhsArrayType.hasStaticShape() || !rhsArrayType.hasStaticShape() || !resultArrayType.hasStaticShape()) {
            return true;
          }

          if (lhsArrayType.getElementType() != rhsArrayType.getElementType() || lhsArrayType.getElementType() != resultArrayType.getElementType()) {
            return true;
          }

          return false;
        });

        target.addDynamicallyLegalOp<SubOp>([](SubOp op) {
          mlir::Type lhsType = op.getLhs().getType();
          mlir::Type rhsType = op.getRhs().getType();

          if (!lhsType.isa<ArrayType>() || !rhsType.isa<ArrayType>()) {
            return true;
          }

          auto lhsArrayType = lhsType.cast<ArrayType>();
          auto rhsArrayType = rhsType.cast<ArrayType>();
          auto resultArrayType = op.getResult().getType().cast<ArrayType>();

          if (!lhsArrayType.hasStaticShape() || !rhsArrayType.hasStaticShape() || !resultArrayType.hasStaticShape()) {
            return true;
          }

          if (lhsArrayType.getElementType() != rhsArrayType.getElementType() || lhsArrayType.getElementType() != resultArrayType.getElementType()) {
            return true;
          }

          return false;
        });

        target.addDynamicallyLegalOp<TransposeOp>([](TransposeOp op) {
          mlir::Type argType = op.getMatrix().getType().cast<ArrayType>().getElementType();
          mlir::Type resultType = op.getResult().getType().cast<ArrayType>().getElementType();
          return argType != resultType;
        });

        target.addIllegalOp<SumOp>();
        target.addIllegalOp<ProductOp>();

        TypeConverter typeConverter(bitWidth);

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToVectorPatterns(patterns, &getContext(), typeConverter);

        return applyPartialConversion(module, target, std::move(patterns));
      }
  };
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToVectorConversionPass()
  {
    return std::make_unique<ModelicaToVectorConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToVectorConversionPass(const ModelicaToVectorConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToVectorConversionPass>(options);
  }
}
