#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Runtime.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
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
      ModelicaOpRewritePattern(mlir::MLIRContext* ctx, ModelicaToLLVMOptions options)
          : mlir::OpRewritePattern<Op>(ctx),
            options(std::move(options))
      {
      }
    
    protected:
      ModelicaToLLVMOptions options;
  };

  /// Generic conversion pattern that provides some utility functions.
  template<typename Op>
  class ModelicaOpConversionPattern : public mlir::ConvertOpToLLVMPattern<Op>
  {
    public:
      ModelicaOpConversionPattern(mlir::MLIRContext* ctx, mlir::LLVMTypeConverter& typeConverter, ModelicaToLLVMOptions options)
          : mlir::ConvertOpToLLVMPattern<Op>(typeConverter, 1),
            options(std::move(options))
      {
      }

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

    protected:
      ModelicaToLLVMOptions options;
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

//===----------------------------------------------------------------------===//
// Various operations
//===----------------------------------------------------------------------===//

namespace
{
  struct AssignmentOpScalarLowering : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!isNumeric(op.getValue())) {
        return rewriter.notifyMatchFailure(op, "Source value has not a numeric type");
      }

      auto destinationBaseType = op.getDestination().getType().cast<ArrayType>().getElementType();
      mlir::Value value = rewriter.create<CastOp>(loc, destinationBaseType, op.getValue());
      rewriter.replaceOpWithNewOp<StoreOp>(op, value, op.getDestination(), llvm::None);

      return mlir::success();
    }
  };

  struct AssignmentOpArrayLowering : public ModelicaOpRewritePattern<AssignmentOp>
  {
    using ModelicaOpRewritePattern<AssignmentOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.getValue().getType().isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source value is not an array");
      }

      iterateArray(rewriter, op.getLoc(), op.getValue(),
                   [&](mlir::OpBuilder& nestedBuilder, mlir::Location, mlir::ValueRange position) {
                     mlir::Value value = rewriter.create<LoadOp>(loc, op.getValue(), position);
                     value = rewriter.create<CastOp>(value.getLoc(), op.getDestination().getType().cast<ArrayType>().getElementType(), value);
                     rewriter.create<StoreOp>(loc, value, op.getDestination(), position);
                   });

      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
}

static void populateModelicaToLLVMPatterns(
		mlir::RewritePatternSet& patterns,
		mlir::MLIRContext* context,
		mlir::LLVMTypeConverter& typeConverter,
		ModelicaToLLVMOptions options)
{
  // Cast operations
  patterns.insert<
      CastOpIntegerLowering,
      CastOpFloatLowering>(context, typeConverter, options);

  // Runtime functions operations
  patterns.insert<
      CallOpLowering, RuntimeFunctionOpLowering>(context, typeConverter, options);

  // Various operations
  patterns.insert<
      AssignmentOpScalarLowering,
      AssignmentOpArrayLowering>(context, options);
}

namespace
{
  class ModelicaToLLVMConversionPass : public ModelicaToLLVMBase<ModelicaToLLVMConversionPass>
  {
    public:
      ModelicaToLLVMConversionPass(ModelicaToLLVMOptions options)
          : options(std::move(options))
      {
      }

      void runOnOperation() override
      {
        if (mlir::failed(convertOperations())) {
          mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica operations");
          return signalPassFailure();
        }

        getOperation().dump();
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<
            CastOp,
            AssignmentOp,
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
        llvmLoweringOptions.dataLayout = options.dataLayout;
        mlir::modelica::LLVMTypeConverter typeConverter(&getContext(), llvmLoweringOptions, options.bitWidth);

        mlir::RewritePatternSet patterns(&getContext());

        populateModelicaToLLVMPatterns(patterns, &getContext(), typeConverter, options);
        populateIDAStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

        return applyPartialConversion(module, target, std::move(patterns));
      }

    private:
      ModelicaToLLVMOptions options;
  };
}

namespace marco::codegen
{
  const ModelicaToLLVMOptions& ModelicaToLLVMOptions::getDefaultOptions()
  {
    static ModelicaToLLVMOptions options;
    return options;
  }

  std::unique_ptr<mlir::Pass> createModelicaToLLVMPass(ModelicaToLLVMOptions options)
  {
    return std::make_unique<ModelicaToLLVMConversionPass>(options);
  }
}
