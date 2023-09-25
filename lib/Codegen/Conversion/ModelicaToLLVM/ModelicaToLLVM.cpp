#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
#define GEN_PASS_DEF_MODELICATOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::marco;
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

        auto ptrType = mlir::LLVM::LLVMPointerType::get(value.getType());

        auto allocated = builder.create<mlir::LLVM::AllocaOp>(
            loc, ptrType, mlir::ValueRange{one});

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

static void populateModelicaToLLVMPatterns(
		mlir::RewritePatternSet& patterns,
		mlir::LLVMTypeConverter& typeConverter)
{
  // Range operations.
  patterns.insert<
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
  class ModelicaToLLVMConversionPass : public mlir::impl::ModelicaToLLVMConversionPassBase<ModelicaToLLVMConversionPass>
  {
    public:
      using ModelicaToLLVMConversionPassBase::ModelicaToLLVMConversionPassBase;

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

        if (mlir::failed(legalizeSimulation())) {
          mlir::emitError(getOperation().getLoc())
              << "Legalization of Simulation dialect failed";

          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertCallOps();

      mlir::LogicalResult convertOperations();

      mlir::LogicalResult legalizeSimulation();
  };
}

mlir::LogicalResult ModelicaToLLVMConversionPass::convertCallOps()
{
  auto moduleOp = getOperation();
  mlir::SymbolTableCollection symbolTable;
  mlir::ConversionTarget target(getContext());

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);
  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<CallOpLowering>(typeConverter);

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

mlir::LogicalResult ModelicaToLLVMConversionPass::convertOperations()
{
  auto moduleOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<
      RangeBeginOp,
      RangeEndOp,
      RangeStepOp>();

  target.addIllegalOp<
      DerOp,
      CastOp>();

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::modelica::LLVMTypeConverter typeConverter(
      &getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());

  populateModelicaToLLVMPatterns(patterns, typeConverter);

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  return applyPartialConversion(moduleOp, target, std::move(patterns));
}

mlir::LogicalResult ModelicaToLLVMConversionPass::legalizeSimulation()
{
  auto module = getOperation();
  mlir::ConversionTarget target(getContext());

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);

  mlir::modelica::LLVMTypeConverter typeConverter(
      &getContext(), llvmLoweringOptions, bitWidth);

  mlir::RewritePatternSet patterns(&getContext());

  populateModelicaToLLVMPatterns(patterns, typeConverter);

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass()
  {
    return std::make_unique<ModelicaToLLVMConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass(
      const ModelicaToLLVMConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToLLVMConversionPass>(options);
  }
}
