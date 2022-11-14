#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/KINSOLToLLVM.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

//===----------------------------------------------------------------------===//
// Cast operations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Runtime functions operations
//===----------------------------------------------------------------------===//

namespace
{
  template<typename Op>
  class RuntimeOpConversionPattern : public ModelicaOpConversionPattern<Op>
  {
    public:
      using ModelicaOpConversionPattern<Op>::ModelicaOpConversionPattern;

    protected:
      const RuntimeFunctionsMangling* getMangler() const
      {
        return &mangler;
      }

      std::string getMangledType(mlir::Type type) const
      {
        if (auto indexType = type.dyn_cast<mlir::IndexType>()) {
          return getMangledType(this->getTypeConverter()->convertType(type));
        }

        if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
          return getMangler()->getIntegerType(integerType.getWidth());
        }

        if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
          return getMangler()->getFloatingPointType(floatType.getWidth());
        }

        if (auto memRefType = type.dyn_cast<mlir::UnrankedMemRefType>()) {
          return getMangler()->getArrayType(getMangledType(memRefType.getElementType()));
        }

        llvm_unreachable("Unknown type for mangling");
        return "unknown";
      }

      mlir::Value promote(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value) const
      {
        auto one = builder.create<mlir::LLVM::ConstantOp>(
            loc, this->getIndexType(), builder.getIndexAttr(1));

        auto ptrType = mlir::LLVM::LLVMPointerType::get(value.getType());
        auto allocated = builder.create<mlir::LLVM::AllocaOp>(loc, ptrType, mlir::ValueRange{one});
        builder.create<mlir::LLVM::StoreOp>(loc, value, allocated);

        return allocated;
      }

    private:
      RuntimeFunctionsMangling mangler;
  };

  struct CallOpLowering : public RuntimeOpConversionPattern<CallOp>
  {
    using RuntimeOpConversionPattern<CallOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Type, 1> resultTypes;

      if (auto res = getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes); mlir::failed(res)) {
        return res;
      }

      llvm::SmallVector<mlir::Value, 3> args;

      for (const auto& arg : adaptor.getArgs()) {
        if (arg.getType().isa<mlir::LLVM::LLVMStructType>()) {
          args.push_back(promote(rewriter, loc, arg));
        } else {
          args.push_back(arg);
        }
      }

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
          op, resultTypes, getMangledFunctionName(op), args);

      return mlir::success();
    }

    std::string getMangledFunctionName(CallOp op) const
    {
      llvm::SmallVector<std::string, 2> mangledArgTypes;

      for (const auto& type : op.getArgs().getTypes()) {
        mangledArgTypes.push_back(getMangledType(type));
      }

      auto resultTypes = op.getResultTypes();
      assert(resultTypes.size() <= 1);

      if (resultTypes.empty()) {
        return getMangler()->getMangledFunction(
            op.getCallee(), getMangler()->getVoidType(), mangledArgTypes);
      }

      return getMangler()->getMangledFunction(
          op.getCallee(), getMangledType(resultTypes[0]), mangledArgTypes);
    }
  };

  class RuntimeFunctionOpLowering : public RuntimeOpConversionPattern<RuntimeFunctionOp>
  {
    using RuntimeOpConversionPattern<RuntimeFunctionOp>::RuntimeOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(RuntimeFunctionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      mlir::Type functionType;
      bool resultPromoted;

      std::tie(functionType, resultPromoted) = getTypeConverter()->convertFunctionTypeCWrapper(op.getFunctionType());
      auto mangledName = getMangledFunctionName(op);

      rewriter.replaceOpWithNewOp<mlir::LLVM::LLVMFuncOp>(
          op, mangledName, functionType.cast<mlir::LLVM::LLVMFunctionType>());

      return mlir::success();
    }

    std::string getMangledFunctionName(RuntimeFunctionOp op) const
    {
      llvm::SmallVector<std::string, 2> mangledArgTypes;

      for (const auto& type : op.getArgumentTypes()) {
        mangledArgTypes.push_back(getMangledType(type));
      }

      auto resultTypes = op.getResultTypes();
      assert(resultTypes.size() <= 1);

      if (resultTypes.empty()) {
        return getMangler()->getMangledFunction(
            op.getSymName(), getMangler()->getVoidType(), mangledArgTypes);
      }

      return getMangler()->getMangledFunction(
          op.getSymName(), getMangledType(resultTypes[0]), mangledArgTypes);
    }
  };
}

static void populateModelicaToLLVMPatterns(
		mlir::RewritePatternSet& patterns,
		mlir::LLVMTypeConverter& typeConverter)
{
  // Cast operations
  patterns.insert<
      CastOpIntegerLowering,
      CastOpFloatLowering>(typeConverter);

  // Runtime functions operations
  patterns.insert<
      CallOpLowering,
      RuntimeFunctionOpLowering>(typeConverter);
}

namespace
{
  class ModelicaToLLVMConversionPass : public mlir::impl::ModelicaToLLVMConversionPassBase<ModelicaToLLVMConversionPass>
  {
    public:
      using ModelicaToLLVMConversionPassBase::ModelicaToLLVMConversionPassBase;

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
          return signalPassFailure();
        }
      }

    private:
      mlir::LogicalResult convertOperations();
  };
}

mlir::LogicalResult ModelicaToLLVMConversionPass::convertOperations()
{
  auto module = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<
      DerOp,
      CastOp,
      RuntimeFunctionOp>();

  target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto callee = module.lookupSymbol(op.getCallee());
    return !mlir::isa<RuntimeFunctionOp>(callee);
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
  llvmLoweringOptions.dataLayout.reset(dataLayout);
  mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

        mlir::RewritePatternSet patterns(&getContext());

        populateModelicaToLLVMPatterns(patterns, typeConverter);
        populateIDAStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
        populateKINSOLStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

  return applyPartialConversion(module, target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass()
  {
    return std::make_unique<ModelicaToLLVMConversionPass>();
  }

  std::unique_ptr<mlir::Pass> createModelicaToLLVMConversionPass(const ModelicaToLLVMConversionPassOptions& options)
  {
    return std::make_unique<ModelicaToLLVMConversionPass>(options);
  }
}
