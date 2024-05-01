#include "marco/Codegen/Conversion/BaseModelicaToLLVM/BaseModelicaToLLVM.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/BaseModelicaCommon/Utils.h"
#include "marco/Dialect/BaseModelica/BaseModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Runtime/RuntimeDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_BASEMODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::bmodelica;

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
  class ModelicaOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      using mlir::ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
      }

      void materializeTargetConversion(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& values) const
      {
        for (auto& value : values) {
          value = materializeTargetConversion(builder, value);
        }
      }
  };
}

//===---------------------------------------------------------------------===//
// Range operations
//===---------------------------------------------------------------------===//

namespace
{
  class ConstantOpRangeLowering
      : public ModelicaOpConversionPattern<ConstantOp>
  {
    public:
      using ModelicaOpConversionPattern<ConstantOp>::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          ConstantOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        if (!op.getResult().getType().isa<RangeType>()) {
          return mlir::failure();
        }

        auto structType =
            getTypeConverter()->convertType(op.getResult().getType())
                .cast<mlir::LLVM::LLVMStructType>();

        mlir::Value result =
            rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

        if (auto rangeAttr = op.getValue().dyn_cast<IntegerRangeAttr>()) {
          auto lowerBoundAttr = rewriter.getIntegerAttr(
              structType.getBody()[0], rangeAttr.getLowerBound());

          auto upperBoundAttr = rewriter.getIntegerAttr(
              structType.getBody()[1], rangeAttr.getUpperBound());

          auto stepAttr = rewriter.getIntegerAttr(
              structType.getBody()[2], rangeAttr.getStep());

          mlir::Value lowerBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

          mlir::Value upperBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

          mlir::Value step =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, lowerBound, 0);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, upperBound, 1);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, step, 2);

          rewriter.replaceOp(op, result);
          return mlir::success();
        }

        if (auto rangeAttr = op.getValue().dyn_cast<RealRangeAttr>()) {
          auto lowerBoundAttr = rewriter.getFloatAttr(
              structType.getBody()[0],
              rangeAttr.getLowerBound().convertToDouble());

          auto upperBoundAttr = rewriter.getFloatAttr(
              structType.getBody()[1],
              rangeAttr.getUpperBound().convertToDouble());

          auto stepAttr = rewriter.getFloatAttr(
              structType.getBody()[2],
              rangeAttr.getStep().convertToDouble());

          mlir::Value lowerBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, lowerBoundAttr);

          mlir::Value upperBound =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, upperBoundAttr);

          mlir::Value step =
              rewriter.create<mlir::LLVM::ConstantOp>(loc, stepAttr);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, lowerBound, 0);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, upperBound, 1);

          result = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, structType, result, step, 2);

          rewriter.replaceOp(op, result);
          return mlir::success();
        }

        return mlir::failure();
      }
  };

  class RangeOpLowering
      : public ModelicaOpConversionPattern<RangeOp>
  {
    public:
      using ModelicaOpConversionPattern<RangeOp>::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto structType =
            getTypeConverter()->convertType(op.getResult().getType())
                .cast<mlir::LLVM::LLVMStructType>();

        mlir::Value lowerBound = adaptor.getLowerBound();

        if (mlir::Type requiredType = structType.getBody()[0];
            lowerBound.getType() != requiredType) {
          lowerBound = rewriter.create<CastOp>(loc, requiredType, lowerBound);
        }

        mlir::Value upperBound = adaptor.getUpperBound();

        if (mlir::Type requiredType = structType.getBody()[1];
            upperBound.getType() != requiredType) {
          upperBound = rewriter.create<CastOp>(loc, requiredType, upperBound);
        }

        mlir::Value step = adaptor.getStep();

        if (mlir::Type requiredType = structType.getBody()[2];
            step.getType() != requiredType) {
          step = rewriter.create<CastOp>(loc, requiredType, step);
        }

        mlir::Value result =
            rewriter.create<mlir::LLVM::UndefOp>(loc, structType);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, lowerBound, 0);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, upperBound, 1);

        result = rewriter.create<mlir::LLVM::InsertValueOp>(
            loc, structType, result, step, 2);

        rewriter.replaceOp(op, result);
        return mlir::success();
      }
  };

  struct RangeBeginOpLowering
      : public ModelicaOpConversionPattern<RangeBeginOp>
  {
      using ModelicaOpConversionPattern<RangeBeginOp>
          ::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeBeginOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 0);

        return mlir::success();
      }
  };

  struct RangeEndOpLowering
      : public ModelicaOpConversionPattern<RangeEndOp>
  {
      using ModelicaOpConversionPattern<RangeEndOp>
          ::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeEndOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 1);

        return mlir::success();
      }
  };

  struct RangeStepOpLowering
      : public ModelicaOpConversionPattern<RangeStepOp>
  {
      using ModelicaOpConversionPattern<RangeStepOp>
          ::ModelicaOpConversionPattern;

      mlir::LogicalResult matchAndRewrite(
          RangeStepOp op,
          OpAdaptor adaptor,
          mlir::ConversionPatternRewriter& rewriter) const override
      {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
            op, adaptor.getRange(), 2);

        return mlir::success();
      }
  };
}

//===---------------------------------------------------------------------===//
// Cast operations
//===---------------------------------------------------------------------===//

namespace
{
  class CastOpIntegerLowering : public ModelicaOpConversionPattern<CastOp>
  {
    using ModelicaOpConversionPattern<CastOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!adaptor.getValue().getType().isa<mlir::IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an IntegerType");
      }

      auto loc = op.getLoc();

      mlir::Type source = adaptor.getValue().getType();
      mlir::Type destination = getTypeConverter()->convertType(op.getResult().getType());

      if (source == destination) {
        rewriter.replaceOp(op, adaptor.getValue());
        return mlir::success();
      }

      if (destination.isa<mlir::IntegerType>()) {
        mlir::Value result = adaptor.getValue();

        if (result.getType().getIntOrFloatBitWidth() < destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::ExtSIOp>(loc, destination, result);
        } else if (result.getType().getIntOrFloatBitWidth() > destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::TruncIOp>(loc, destination, result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      if (destination.isa<mlir::FloatType>()) {
        mlir::Value result = adaptor.getValue();

        if (result.getType().getIntOrFloatBitWidth() < destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::ExtSIOp>(
              loc, rewriter.getIntegerType(destination.getIntOrFloatBitWidth()), result);
        } else if (result.getType().getIntOrFloatBitWidth() > destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::TruncIOp>(
              loc, rewriter.getIntegerType(destination.getIntOrFloatBitWidth()), result);
        }

        result = rewriter.create<mlir::arith::SIToFPOp>(loc, destination, result);
        rewriter.replaceOp(op, result);

        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown cast");
    }
  };

  class CastOpFloatLowering : public ModelicaOpConversionPattern<CastOp>
  {
    using ModelicaOpConversionPattern<CastOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!adaptor.getValue().getType().isa<mlir::FloatType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not a FloatType");
      }

      auto loc = op.getLoc();

      mlir::Type source = adaptor.getValue().getType();
      mlir::Type destination = getTypeConverter()->convertType(op.getResult().getType());

      if (source == destination) {
        rewriter.replaceOp(op, adaptor.getValue());
        return mlir::success();
      }

      if (destination.isa<mlir::IntegerType>()) {
        mlir::Value result = adaptor.getValue();

        result = rewriter.create<mlir::arith::FPToSIOp>(
            loc, rewriter.getIntegerType(result.getType().getIntOrFloatBitWidth()), result);

        if (result.getType().getIntOrFloatBitWidth() < destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::ExtSIOp>(
              loc, rewriter.getIntegerType(destination.getIntOrFloatBitWidth()), result);
        } else if (result.getType().getIntOrFloatBitWidth() > destination.getIntOrFloatBitWidth()) {
          result = rewriter.create<mlir::arith::TruncIOp>(
              loc, rewriter.getIntegerType(destination.getIntOrFloatBitWidth()), result);
        }

        rewriter.replaceOp(op, result);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown cast");
    }
  };
}

//===---------------------------------------------------------------------===//
// Runtime functions operations
//===---------------------------------------------------------------------===//

namespace
{
  template<typename Op>
  class RuntimeOpConversionPattern : public ModelicaOpConversionPattern<Op>
  {
    public:
      using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;

    protected:
      mlir::Value promote(
        mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) const
      {
        auto one = builder.create<mlir::LLVM::ConstantOp>(
            loc, this->getIndexType(), builder.getIndexAttr(1));

        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        auto allocated = builder.create<mlir::LLVM::AllocaOp>(
            loc, ptrType, value.getType(), one);

        builder.create<mlir::LLVM::StoreOp>(loc, value, allocated);

        mlir::Value opaquePtr = builder.create<mlir::LLVM::BitcastOp>(
            loc,
            mlir::LLVM::LLVMPointerType::get(builder.getContext()),
            allocated);

        return opaquePtr;
      }
  };

  struct CallOpLowering : public RuntimeOpConversionPattern<CallOp>
  {
    using RuntimeOpConversionPattern<CallOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        CallOp op,
        OpAdaptor adaptor,
        mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Location loc = op.getLoc();
      llvm::SmallVector<mlir::Type, 1> resultTypes;

      if (mlir::failed(getTypeConverter()->convertTypes(
              op.getResultTypes(), resultTypes))) {
        return mlir::failure();
      }

      llvm::SmallVector<mlir::Value, 3> args;

      for (mlir::Value arg : adaptor.getArgs()) {
        if (arg.getType().isa<mlir::LLVM::LLVMStructType>()) {
          args.push_back(promote(rewriter, loc, arg));
        } else {
          args.push_back(arg);
        }
      }

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, resultTypes, op.getCallee().getLeafReference(), args);

      return mlir::success();
    }
  };
}

static void populateBaseModelicaToLLVMPatterns(
		mlir::RewritePatternSet& patterns,
		mlir::LLVMTypeConverter& typeConverter)
{
  // Range operations.
  patterns.insert<
      ConstantOpRangeLowering,
      RangeOpLowering,
      RangeBeginOpLowering,
      RangeEndOpLowering,
      RangeStepOpLowering>(typeConverter);

  // Cast operations.
  patterns.insert<
      CastOpIntegerLowering,
      CastOpFloatLowering>(typeConverter);
}

namespace
{
  class BaseModelicaToLLVMConversionPass : public mlir::impl::BaseModelicaToLLVMConversionPassBase<BaseModelicaToLLVMConversionPass>
  {
    public:
      using BaseModelicaToLLVMConversionPassBase::BaseModelicaToLLVMConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertCallOps())) {
          mlir::emitError(getOperation().getLoc())
              << "Modelica to LLVM conversion failed";

          return signalPassFailure();
        }

        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc())
              << "Modelica to LLVM conversion failed";

          return signalPassFailure();
        }

        if (mlir::failed(legalizeRuntime())) {
          mlir::emitError(getOperation().getLoc())
              << "Legalization of Runtime dialect failed";

          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertCallOps();

      mlir::LogicalResult convertOperations();

      mlir::LogicalResult legalizeRuntime();
  };
}

mlir::LogicalResult BaseModelicaToLLVMConversionPass::convertCallOps()
{
  auto moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;
  mlir::ConversionTarget target(getContext());

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);
  mlir::bmodelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<CallOpLowering>(typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

mlir::LogicalResult BaseModelicaToLLVMConversionPass::convertOperations()
{
  auto moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addDynamicallyLegalOp<ConstantOp>([](ConstantOp op) {
    return !op.getResult().getType().isa<RangeType>();
  });

  target.addIllegalOp<
      RangeOp,
      RangeBeginOp,
      RangeEndOp,
      RangeStepOp>();

  target.addIllegalOp<
      DerOp,
      CastOp>();

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::bmodelica::LLVMTypeConverter typeConverter(
      &getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());

  populateBaseModelicaToLLVMPatterns(patterns, typeConverter);

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

mlir::LogicalResult BaseModelicaToLLVMConversionPass::legalizeRuntime()
{
  auto module = getOperation();
  mlir::ConversionTarget target(getContext());

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::bmodelica::LLVMTypeConverter typeConverter(
      &getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());

  populateBaseModelicaToLLVMPatterns(patterns, typeConverter);

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass()
  {
    return std::make_unique<BaseModelicaToLLVMConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createBaseModelicaToLLVMConversionPass(
      const BaseModelicaToLLVMConversionPassOptions& options)
  {
    return std::make_unique<BaseModelicaToLLVMConversionPass>(options);
  }
}
