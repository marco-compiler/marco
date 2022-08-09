#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/Utils.h"
#include "marco/Codegen/ArrayDescriptor.h"
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
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"

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

/// Get or declare an LLVM function inside the module.
static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(
    mlir::OpBuilder& builder,
    mlir::ModuleOp module,
    llvm::StringRef name,
    mlir::Type result,
    llvm::ArrayRef<mlir::Type> args)
{
	if (auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    return funcOp;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, mlir::LLVM::LLVMFunctionType::get(result, args));
}

/// Get or declare an LLVM function inside the module.
static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::Type result, mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (const auto& arg : args) {
    argsTypes.push_back(arg.getType());
  }

	return getOrDeclareFunction(builder, module, name, result, argsTypes);
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
      const RuntimeFunctionsMangling* getMangler() const
      {
        return &mangler;
      }

      std::string getMangledType(mlir::Type type) const
      {
        if (auto booleanType = type.dyn_cast<BooleanType>()) {
          return getMangler()->getIntegerType(1);
        }

        if (auto integerType = type.dyn_cast<IntegerType>()) {
          mlir::Type convertedType = this->getTypeConverter()->convertType(integerType);
          return getMangler()->getIntegerType(convertedType.getIntOrFloatBitWidth());
        }

        if (auto realType = type.dyn_cast<RealType>()) {
          mlir::Type convertedType = this->getTypeConverter()->convertType(realType);
          return getMangler()->getFloatingPointType(convertedType.getIntOrFloatBitWidth());
        }

        if (auto arrayType = type.dyn_cast<ArrayType>()) {
          return getMangler()->getArrayType(getMangledType(arrayType.getElementType()));
        }

        if (auto indexType = type.dyn_cast<mlir::IndexType>()) {
          return getMangledType(this->getTypeConverter()->convertType(type));
        }

        if (auto integerType = type.dyn_cast<mlir::IntegerType>()) {
          return getMangler()->getIntegerType(integerType.getWidth());
        }

        if (auto floatType = type.dyn_cast<mlir::FloatType>()) {
          return getMangler()->getFloatingPointType(floatType.getWidth());
        }

        llvm_unreachable("Unknown type for mangling");
        return "unknown";
      }

      mlir::Value convertToUnsizedArray(mlir::OpBuilder& builder, mlir::Value array) const
      {
        auto loc = array.getLoc();
        ArrayDescriptor sourceDescriptor(this->getTypeConverter(), array);

        // Create the unsized array descriptor that holds the ranked one. It is allocated on the stack, because
        // the runtime library expects a pointer in order to avoid any unrolling due to calling conventions.
        // The inner descriptor (that is, the sized array descriptor) is also allocated on the stack.
        mlir::Type indexType = this->getTypeConverter()->getIndexType();
        mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 1));

        // Allocate space on the stack for the sized array descriptor
        auto sizedArrayDescPtrType = mlir::LLVM::LLVMPointerType::get(array.getType());
        mlir::Value sizedArrayDescNullPtr = builder.create<mlir::LLVM::NullOp>(loc, sizedArrayDescPtrType);
        mlir::Value sizedArrayDescGepPtr = builder.create<mlir::LLVM::GEPOp>(loc, sizedArrayDescPtrType, sizedArrayDescNullPtr, one);
        mlir::Value sizedArrayDescSizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(loc, indexType, sizedArrayDescGepPtr);
        mlir::Value sizedArrayDescOpaquePtr = builder.create<mlir::LLVM::AllocaOp>(loc, this->getVoidPtrType(), sizedArrayDescSizeBytes);
        mlir::Value sizedArrayDescPtr = builder.create<mlir::LLVM::BitcastOp>(loc, sizedArrayDescPtrType, sizedArrayDescOpaquePtr);

        // Determine the type of the unsized array descriptor
        llvm::SmallVector<mlir::Type, 3> unsizedArrayStructTypes;

        unsizedArrayStructTypes.push_back(this->getTypeConverter()->getIndexType());
        unsizedArrayStructTypes.push_back(this->getVoidPtrType());

        mlir::Type unsizedArrayDescriptorType = mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), unsizedArrayStructTypes);
        mlir::Type unsizedArrayDescriptorPtrType = mlir::LLVM::LLVMPointerType::get(unsizedArrayDescriptorType);

        // Allocate space on the stack for the unsized array descriptor
        mlir::Value unsizedArrayDescNullPtr = builder.create<mlir::LLVM::NullOp>(loc, unsizedArrayDescriptorPtrType);
        mlir::Value unsizedArrayDescGepPtr = builder.create<mlir::LLVM::GEPOp>(loc, unsizedArrayDescriptorPtrType, unsizedArrayDescNullPtr, one);
        mlir::Value unsizedArrayDescSizeBytes = builder.create<mlir::LLVM::PtrToIntOp>(loc, indexType, unsizedArrayDescGepPtr);
        mlir::Value unsizedArrayDescOpaquePtr = builder.create<mlir::LLVM::AllocaOp>(loc, this->getVoidPtrType(), unsizedArrayDescSizeBytes);
        mlir::Value unsizedArrayDescPtr = builder.create<mlir::LLVM::BitcastOp>(loc, unsizedArrayDescriptorPtrType, unsizedArrayDescOpaquePtr);

        // Populate the sized array descriptor
        builder.create<mlir::LLVM::StoreOp>(loc, *sourceDescriptor, sizedArrayDescPtr);

        // Populate the unsized array descriptor
        UnsizedArrayDescriptor unsizedArrayDescriptor = UnsizedArrayDescriptor::undef(builder, loc, unsizedArrayDescriptorType);
        unsizedArrayDescriptor.setPtr(builder, loc, sizedArrayDescOpaquePtr);
        unsizedArrayDescriptor.setRank(builder, loc, sourceDescriptor.getRank(builder, loc));
        builder.create<mlir::LLVM::StoreOp>(loc, *unsizedArrayDescriptor, unsizedArrayDescPtr);

        return unsizedArrayDescPtr;
      }

    protected:
      ModelicaToLLVMOptions options;

    private:
      RuntimeFunctionsMangling mangler;
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
// Math operations
//===----------------------------------------------------------------------===//

namespace
{
  struct PowOpLowering: public ModelicaOpConversionPattern<PowOp>
  {
    using ModelicaOpConversionPattern<PowOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(PowOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!isNumeric(op.getBase())) {
        return rewriter.notifyMatchFailure(op, "Base is not a scalar");
      }

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Base
      newOperands.push_back(adaptor.getBase());
      mangledArgsTypes.push_back(getMangledType(op.getBase().getType()));

      // Exponent
      newOperands.push_back(adaptor.getExponent());
      mangledArgsTypes.push_back(getMangledType(op.getExponent().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("pow", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };
}

//===----------------------------------------------------------------------===//
// Built-in functions
//===----------------------------------------------------------------------===//

namespace
{
  struct AbsOpCastPattern : public ModelicaOpRewritePattern<AbsOp>
  {
    using ModelicaOpRewritePattern<AbsOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AbsOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType != resultType);
    }

    void rewrite(AbsOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AbsOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AbsOpLowering : public ModelicaOpConversionPattern<AbsOp>
  {
    using ModelicaOpConversionPattern<AbsOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AbsOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(AbsOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("abs", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct AcosOpCastPattern : public ModelicaOpRewritePattern<AcosOp>
  {
    using ModelicaOpRewritePattern<AcosOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AcosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AcosOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AcosOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AcosOpLowering : public ModelicaOpConversionPattern<AcosOp>
  {
    using ModelicaOpConversionPattern<AcosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AcosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(AcosOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("acos", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct AsinOpCastPattern : public ModelicaOpRewritePattern<AsinOp>
  {
    using ModelicaOpRewritePattern<AsinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AsinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AsinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AsinOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AsinOpLowering : public ModelicaOpConversionPattern<AsinOp>
  {
    using ModelicaOpConversionPattern<AsinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AsinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(AsinOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("asin", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct AtanOpCastPattern : public ModelicaOpRewritePattern<AtanOp>
  {
    using ModelicaOpRewritePattern<AtanOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(AtanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(AtanOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<AtanOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct AtanOpLowering : public ModelicaOpConversionPattern<AtanOp>
  {
    using ModelicaOpConversionPattern<AtanOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(AtanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(AtanOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("atan", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct Atan2OpCastPattern : public ModelicaOpRewritePattern<Atan2Op>
  {
    using ModelicaOpRewritePattern<Atan2Op>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(Atan2Op op) const override
    {
      mlir::Type yType = op.getY().getType();
      mlir::Type xType = op.getX().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !yType.isa<RealType>() || !xType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(Atan2Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());

      mlir::Value y = rewriter.create<CastOp>(loc, realType, op.getY());
      mlir::Value x = rewriter.create<CastOp>(loc, realType, op.getX());

      mlir::Value result = rewriter.create<Atan2Op>(loc, realType, y, x);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct Atan2OpLowering : public ModelicaOpConversionPattern<Atan2Op>
  {
    using ModelicaOpConversionPattern<Atan2Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(Atan2Op op) const override
    {
      mlir::Type yType = op.getY().getType();
      mlir::Type xType = op.getX().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          yType.isa<RealType>() && xType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(Atan2Op op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Y
      assert(op.getY().getType().isa<RealType>());
      newOperands.push_back(adaptor.getY());
      mangledArgsTypes.push_back(getMangledType(op.getY().getType()));

      // X
      assert(op.getX().getType().isa<RealType>());
      newOperands.push_back(adaptor.getX());
      mangledArgsTypes.push_back(getMangledType(op.getX().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("atan2", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct CeilOpCastPattern : public ModelicaOpRewritePattern<CeilOp>
  {
    using ModelicaOpRewritePattern<CeilOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CeilOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(CeilOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<CeilOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct CeilOpLowering : public ModelicaOpConversionPattern<CeilOp>
  {
    using ModelicaOpConversionPattern<CeilOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(CeilOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(CeilOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("ceil", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct CosOpCastPattern : public ModelicaOpRewritePattern<CosOp>
  {
    using ModelicaOpRewritePattern<CosOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    mlir::LogicalResult matchAndRewrite(CosOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<CosOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct CosOpLowering : public ModelicaOpConversionPattern<CosOp>
  {
    using ModelicaOpConversionPattern<CosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(CosOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(CosOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("cos", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct CoshOpCastPattern : public ModelicaOpRewritePattern<CoshOp>
  {
    using ModelicaOpRewritePattern<CoshOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(CoshOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(CoshOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<CoshOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct CoshOpLowering : public ModelicaOpConversionPattern<CoshOp>
  {
    using ModelicaOpConversionPattern<CoshOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(CoshOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(CoshOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("cosh", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct DiagonalOpLowering : public ModelicaOpConversionPattern<DiagonalOp>
  {
    using ModelicaOpConversionPattern<DiagonalOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DiagonalOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          if (dynamicDimensions.empty()) {
            assert(op.getValues().getType().cast<ArrayType>().getRank() == 1);
            mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

            mlir::Value values = getTypeConverter()->materializeSourceConversion(
                rewriter, op.getValues().getLoc(), op.getValues().getType(), adaptor.getValues());

            dynamicDimensions.push_back(rewriter.create<DimOp>(loc, values, zeroValue));
          } else {
            dynamicDimensions.push_back(dynamicDimensions[0]);
          }
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Operand
      assert(op.getOperand().getType().isa<ArrayType>());
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getValues()));
      mangledArgsTypes.push_back(getMangledType(op.getValues().getType()));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("diagonal", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct DivTruncOpCastPattern : public ModelicaOpRewritePattern<DivTruncOp>
  {
    using ModelicaOpRewritePattern<DivTruncOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(DivTruncOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(DivTruncOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<DivTruncOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct DivTruncOpLowering : public ModelicaOpConversionPattern<DivTruncOp>
  {
    using ModelicaOpConversionPattern<DivTruncOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(DivTruncOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(DivTruncOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      assert(op.getX().getType() == op.getY().getType());

      // Dividend
      newOperands.push_back(adaptor.getX());
      mangledArgsTypes.push_back(getMangledType(op.getX().getType()));

      // Divisor
      newOperands.push_back(adaptor.getY());
      mangledArgsTypes.push_back(getMangledType(op.getY().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("div", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct ExpOpCastPattern : public ModelicaOpRewritePattern<ExpOp>
  {
    using ModelicaOpRewritePattern<ExpOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ExpOp op) const override
    {
      mlir::Type operandType = op.getExponent().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(ExpOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value exponent = rewriter.create<CastOp>(loc, realType, op.getExponent());
      mlir::Value result = rewriter.create<ExpOp>(loc, realType, exponent);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct ExpOpLowering : public ModelicaOpConversionPattern<ExpOp>
  {
    using ModelicaOpConversionPattern<ExpOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(ExpOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(ExpOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Exponent
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getExponent());
      mangledArgsTypes.push_back(getMangledType(op.getExponent().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("exp", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct FloorOpCastPattern : public ModelicaOpRewritePattern<FloorOp>
  {
    using ModelicaOpRewritePattern<FloorOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(FloorOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(FloorOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<FloorOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct FloorOpLowering : public ModelicaOpConversionPattern<FloorOp>
  {
    using ModelicaOpConversionPattern<FloorOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(FloorOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(FloorOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("floor", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct IdentityOpCastPattern : public ModelicaOpRewritePattern<IdentityOp>
  {
    using ModelicaOpRewritePattern<IdentityOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(IdentityOp op) const override
    {
      mlir::Type sizeType = op.getSize().getType();
      return mlir::LogicalResult::success(!sizeType.isa<IntegerType>());
    }

    void rewrite(IdentityOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto integerType = IntegerType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, integerType, op.getSize());
      rewriter.replaceOpWithNewOp<IdentityOp>(op, op.getResult().getType(), operand);
    }
  };

  struct IdentityOpLowering : public ModelicaOpConversionPattern<IdentityOp>
  {
    using ModelicaOpConversionPattern<IdentityOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(IdentityOp op) const override
    {
      mlir::Type sizeType = op.getSize().getType();
      return mlir::LogicalResult::success(sizeType.isa<IntegerType>());
    }

    void rewrite(IdentityOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          if (dynamicDimensions.empty()) {
            mlir::Value dimensionSize = getTypeConverter()->materializeSourceConversion(
                rewriter, op.getSize().getLoc(), op.getSize().getType(), adaptor.getSize());

            dynamicDimensions.push_back(dimensionSize);
          } else {
            dynamicDimensions.push_back(dynamicDimensions[0]);
          }
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("identity", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
    }
  };

  struct IntegerOpCastPattern : public ModelicaOpRewritePattern<IntegerOp>
  {
    using ModelicaOpRewritePattern<IntegerOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(IntegerOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(IntegerOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      mlir::Value result = rewriter.create<IntegerOp>(loc, op.getOperand().getType(), op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct IntegerOpLowering : public ModelicaOpConversionPattern<IntegerOp>
  {
    using ModelicaOpConversionPattern<IntegerOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(IntegerOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(operandType == resultType);
    }

    void rewrite(IntegerOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("integer", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct LinspaceOpCastPattern : public ModelicaOpRewritePattern<LinspaceOp>
  {
    using ModelicaOpRewritePattern<LinspaceOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(LinspaceOp op) const override
    {
      mlir::Type beginType = op.getBegin().getType();
      mlir::Type endType = op.getEnd().getType();
      mlir::Type amountType = op.getAmount().getType();

      return mlir::LogicalResult::success(
          !beginType.isa<RealType>() || !endType.isa<RealType>() || !amountType.isa<IntegerType>());
    }

    void rewrite(LinspaceOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto integerType = IntegerType::get(op.getContext());
      auto realType = RealType::get(op.getContext());

      mlir::Value begin = rewriter.create<CastOp>(loc, realType, op.getBegin());
      mlir::Value end = rewriter.create<CastOp>(loc, realType, op.getEnd());
      mlir::Value amount = rewriter.create<CastOp>(loc, integerType, op.getAmount());

      rewriter.replaceOpWithNewOp<LinspaceOp>(op, op.getResult().getType(), begin, end, amount);
    }
  };

  struct LinspaceOpLowering : public ModelicaOpConversionPattern<LinspaceOp>
  {
    using ModelicaOpConversionPattern<LinspaceOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(LinspaceOp op) const override
    {
      mlir::Type beginType = op.getBegin().getType();
      mlir::Type endType = op.getEnd().getType();
      mlir::Type amountType = op.getAmount().getType();

      return mlir::LogicalResult::success(
          beginType.isa<RealType>() && endType.isa<RealType>() && amountType.isa<IntegerType>());
    }

    void rewrite(LinspaceOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 3> newOperands;
      llvm::SmallVector<std::string, 3> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 1);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          mlir::Value dimensionSize = getTypeConverter()->materializeSourceConversion(
              rewriter, op.getAmount().getLoc(), op.getAmount().getType(), adaptor.getAmount());

          dimensionSize = rewriter.create<CastOp>(dimensionSize.getLoc(), rewriter.getIndexType(), dimensionSize);
          dynamicDimensions.push_back(dimensionSize);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Begin value
      newOperands.push_back(adaptor.getBegin());
      mangledArgsTypes.push_back(getMangledType(op.getBegin().getType()));

      // End value
      newOperands.push_back(adaptor.getEnd());
      mangledArgsTypes.push_back(getMangledType(op.getEnd().getType()));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("linspace", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
    }
  };

  struct LogOpCastPattern : public ModelicaOpRewritePattern<LogOp>
  {
    using ModelicaOpRewritePattern<LogOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(LogOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(LogOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<LogOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct LogOpLowering : public ModelicaOpConversionPattern<LogOp>
  {
    using ModelicaOpConversionPattern<LogOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(LogOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(LogOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("log", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct Log10OpCastPattern : public ModelicaOpRewritePattern<Log10Op>
  {
    using ModelicaOpRewritePattern<Log10Op>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(Log10Op op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(Log10Op op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<Log10Op>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct Log10OpLowering : public ModelicaOpConversionPattern<Log10Op>
  {
    using ModelicaOpConversionPattern<Log10Op>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(Log10Op op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(Log10Op op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("log10", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct OnesOpLowering : public ModelicaOpConversionPattern<OnesOp>
  {
    using ModelicaOpConversionPattern<OnesOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(OnesOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          auto index = dynamicDimensions.size();
          mlir::Value dimensionSize = adaptor.getSizes()[index];

          dimensionSize = getTypeConverter()->materializeSourceConversion(
              rewriter, dimensionSize.getLoc(), op.getSizes()[index].getType(), adaptor.getSizes()[index]);

          dimensionSize = rewriter.create<CastOp>(dimensionSize.getLoc(), rewriter.getIndexType(), dimensionSize);
          dynamicDimensions.push_back(dimensionSize);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("ones", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct MaxOpArrayCastPattern : public ModelicaOpRewritePattern<MaxOp>
  {
    using ModelicaOpRewritePattern<MaxOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(MaxOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto elementType = op.getFirst().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<MaxOp>(loc, elementType, op.getFirst());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MaxOpArrayLowering : public ModelicaOpConversionPattern<MaxOp>
  {
    using ModelicaOpConversionPattern<MaxOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() == resultType);
    }

    void rewrite(MaxOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      assert(op.getFirst().getType().isa<ArrayType>());

      // Array
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getFirst()));
      mangledArgsTypes.push_back(getMangledType(op.getFirst().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("max", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct MaxOpScalarsCastPattern : public ModelicaOpRewritePattern<MaxOp>
  {
    using ModelicaOpRewritePattern<MaxOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType != secondValueType || firstValueType != resultType);
    }

    void rewrite(MaxOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getFirst(), op.getSecond() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<MaxOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MaxOpScalarsLowering : public ModelicaOpConversionPattern<MaxOp>
  {
    using ModelicaOpConversionPattern<MaxOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(MaxOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType == secondValueType && firstValueType == resultType);
    }

    void rewrite(MaxOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      assert(op.getFirst().getType() == op.getSecond().getType());

      // First value
      newOperands.push_back(adaptor.getFirst());
      mangledArgsTypes.push_back(getMangledType(op.getFirst().getType()));

      // Second value
      newOperands.push_back(adaptor.getSecond());
      mangledArgsTypes.push_back(getMangledType(op.getSecond().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("max", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct MinOpArrayCastPattern : public ModelicaOpRewritePattern<MinOp>
  {
    using ModelicaOpRewritePattern<MinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() != resultType);
    }

    void rewrite(MinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto elementType = op.getFirst().getType().cast<ArrayType>().getElementType();
      mlir::Value result = rewriter.create<MinOp>(loc, elementType, op.getFirst());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MinOpArrayLowering : public ModelicaOpConversionPattern<MinOp>
  {
    using ModelicaOpConversionPattern<MinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 1) {
        return mlir::failure();
      }

      auto arrayType = op.getFirst().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() == resultType);
    }

    void rewrite(MinOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      assert(op.getFirst().getType().isa<ArrayType>());

      // Array
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getFirst()));
      mangledArgsTypes.push_back(getMangledType(op.getFirst().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("min", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct MinOpScalarsCastPattern : public ModelicaOpRewritePattern<MinOp>
  {
    using ModelicaOpRewritePattern<MinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType != secondValueType || firstValueType != resultType);
    }

    void rewrite(MinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getFirst(), op.getSecond() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<MinOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct MinOpScalarsLowering : public ModelicaOpConversionPattern<MinOp>
  {
    using ModelicaOpConversionPattern<MinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(MinOp op) const override
    {
      if (op.getNumOperands() != 2) {
        return mlir::failure();
      }

      mlir::Type firstValueType = op.getFirst().getType();
      mlir::Type secondValueType = op.getSecond().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          firstValueType == secondValueType && firstValueType == resultType);
    }

    void rewrite(MinOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      assert(op.getFirst().getType() == op.getSecond().getType());

      // First value
      newOperands.push_back(adaptor.getFirst());
      mangledArgsTypes.push_back(getMangledType(op.getFirst().getType()));

      // Second value
      newOperands.push_back(adaptor.getSecond());
      mangledArgsTypes.push_back(getMangledType(op.getSecond().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("min", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct ModOpCastPattern : public ModelicaOpRewritePattern<ModOp>
  {
    using ModelicaOpRewritePattern<ModOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(ModOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(ModOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<ModOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct ModOpLowering : public ModelicaOpConversionPattern<ModOp>
  {
    using ModelicaOpConversionPattern<ModOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(ModOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(ModOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      assert(op.getX().getType() == op.getY().getType());

      // Dividend
      newOperands.push_back(adaptor.getX());
      mangledArgsTypes.push_back(getMangledType(op.getX().getType()));

      // Divisor
      newOperands.push_back(adaptor.getY());
      mangledArgsTypes.push_back(getMangledType(op.getY().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("mod", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct NDimsOpLowering : public ModelicaOpRewritePattern<NDimsOp>
  {
    using ModelicaOpRewritePattern<NDimsOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(NDimsOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct ProductOpCastPattern : public ModelicaOpRewritePattern<ProductOp>
  {
    using ModelicaOpRewritePattern<ProductOp>::ModelicaOpRewritePattern;

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

  struct ProductOpLowering : public ModelicaOpConversionPattern<ProductOp>
  {
    using ModelicaOpConversionPattern<ProductOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(ProductOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() == resultType);
    }

    void rewrite(ProductOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Array
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getArray()));
      mangledArgsTypes.push_back(getMangledType(op.getArray().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("product", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct RemOpCastPattern : public ModelicaOpRewritePattern<RemOp>
  {
    using ModelicaOpRewritePattern<RemOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(RemOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType != yType || xType != resultType);
    }

    void rewrite(RemOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      llvm::SmallVector<mlir::Value, 2> castedValues;
      castToMostGenericType(rewriter, llvm::makeArrayRef({ op.getX(), op.getY() }), castedValues);
      assert(castedValues[0].getType() == castedValues[1].getType());
      mlir::Value result = rewriter.create<RemOp>(loc, castedValues[0].getType(), castedValues[0], castedValues[1]);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct RemOpLowering : public ModelicaOpConversionPattern<RemOp>
  {
    using ModelicaOpConversionPattern<RemOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(RemOp op) const override
    {
      mlir::Type xType = op.getX().getType();
      mlir::Type yType = op.getY().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          xType == yType && xType == resultType);
    }

    void rewrite(RemOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      assert(op.getX().getType() == op.getY().getType());

      // 'x' value
      newOperands.push_back(adaptor.getX());
      mangledArgsTypes.push_back(getMangledType(op.getX().getType()));

      // 'y' value
      newOperands.push_back(adaptor.getY());
      mangledArgsTypes.push_back(getMangledType(op.getY().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("rem", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SignOpCastPattern : public ModelicaOpRewritePattern<SignOp>
  {
    using ModelicaOpRewritePattern<SignOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SignOp op) const override
    {
      mlir::Type resultType = op.getResult().getType();
      return mlir::LogicalResult::success(!resultType.isa<IntegerType>());
    }

    void rewrite(SignOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto integerType = IntegerType::get(op.getContext());
      mlir::Value result = rewriter.create<SignOp>(loc, integerType, op.getOperand());
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SignOpLowering : public ModelicaOpConversionPattern<SignOp>
  {
    using ModelicaOpConversionPattern<SignOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(SignOp op) const override
    {
      mlir::Type resultType = op.getResult().getType();
      return mlir::LogicalResult::success(resultType.isa<IntegerType>());
    }

    void rewrite(SignOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<IntegerType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("sign", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SinOpCastPattern : public ModelicaOpRewritePattern<SinOp>
  {
    using ModelicaOpRewritePattern<SinOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SinOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SinOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SinOpLowering : public ModelicaOpConversionPattern<SinOp>
  {
    using ModelicaOpConversionPattern<SinOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(SinOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(SinOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("sin", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SinhOpCastPattern : public ModelicaOpRewritePattern<SinhOp>
  {
    using ModelicaOpRewritePattern<SinhOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SinhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SinhOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SinhOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SinhOpLowering : public ModelicaOpConversionPattern<SinhOp>
  {
    using ModelicaOpConversionPattern<SinhOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(SinhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(SinhOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("sinh", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SizeOpDimensionLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      if (!op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "No index specified");
      }

      mlir::Value index = rewriter.create<CastOp>(loc, rewriter.getIndexType(), op.getDimension());
      mlir::Value result = rewriter.create<DimOp>(loc, op.getArray(), index);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
      return mlir::success();
    }
  };

  struct SizeOpArrayLowering : public ModelicaOpRewritePattern<SizeOp>
  {
    using ModelicaOpRewritePattern<SizeOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(SizeOp op, mlir::PatternRewriter& rewriter) const override
    {
      mlir::Location loc = op->getLoc();

      if (op.hasDimension()) {
        return rewriter.notifyMatchFailure(op, "Index specified");
      }

      assert(op.getResult().getType().isa<ArrayType>());
      auto resultArrayType = op.getResult().getType().cast<ArrayType>();
      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultArrayType, llvm::None);

      // Iterate on each dimension
      mlir::Value zeroValue = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Value rank = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
      mlir::Value step = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

      auto loop = rewriter.create<mlir::scf::ForOp>(loc, zeroValue, rank, step);

      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());

        // Get the size of the current dimension
        mlir::Value dimensionSize = rewriter.create<SizeOp>(loc, resultArrayType.getElementType(), op.getArray(), loop.getInductionVar());

        // Store it into the result array
        rewriter.create<StoreOp>(loc, dimensionSize, result, loop.getInductionVar());
      }

      return mlir::success();
    }
  };

  struct SqrtOpCastPattern : public ModelicaOpRewritePattern<SqrtOp>
  {
    using ModelicaOpRewritePattern<SqrtOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(SqrtOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(SqrtOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<SqrtOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct SqrtOpLowering : public ModelicaOpConversionPattern<SqrtOp>
  {
    using ModelicaOpConversionPattern<SqrtOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(SqrtOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(SqrtOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("sqrt", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SumOpCastPattern : public ModelicaOpRewritePattern<SumOp>
  {
    using ModelicaOpRewritePattern<SumOp>::ModelicaOpRewritePattern;

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

  struct SumOpLowering : public ModelicaOpConversionPattern<SumOp>
  {
    using ModelicaOpConversionPattern<SumOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(SumOp op) const override
    {
      auto arrayType = op.getArray().getType().cast<ArrayType>();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(arrayType.getElementType() == resultType);
    }

    void rewrite(SumOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Array
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getArray()));
      mangledArgsTypes.push_back(getMangledType(op.getArray().getType()));

      // Create the call to the runtime library
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("sum", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct SymmetricOpLowering : public ModelicaOpConversionPattern<SymmetricOp>
  {
    using ModelicaOpConversionPattern<SymmetricOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SymmetricOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      mlir::Value sourceMatrixValue = nullptr;

      auto sourceMatrixFn = [&]() -> mlir::Value {
        if (sourceMatrixValue == nullptr) {
          sourceMatrixValue = getTypeConverter()->materializeSourceConversion(
              rewriter, op.getMatrix().getLoc(), op.getMatrix().getType(), adaptor.getMatrix());
        }

        return sourceMatrixValue;
      };

      if (options.assertions) {
        // Check if the matrix is a square one
        if (!op.getMatrix().getType().cast<ArrayType>().hasStaticShape()) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
          mlir::Value lhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, sourceMatrixFn(), one));
          mlir::Value rhsDimensionSize = materializeTargetConversion(rewriter, rewriter.create<DimOp>(loc, sourceMatrixFn(), zero));
          mlir::Value condition = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhsDimensionSize, rhsDimensionSize);
          rewriter.create<mlir::cf::AssertOp>(loc, condition, rewriter.getStringAttr("Base matrix is not squared"));
        }
      }

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      if (!resultType.hasStaticShape()) {
        for (const auto& dimension : llvm::enumerate(resultType.getShape())) {
          if (dimension.value() == ArrayType::kDynamicSize) {
            mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(dimension.index()));
            dynamicDimensions.push_back(rewriter.create<DimOp>(loc, sourceMatrixFn(), one));
          }
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Matrix
      assert(op.getMatrix().getType().isa<ArrayType>());
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getMatrix()));
      mangledArgsTypes.push_back(getMangledType(op.getMatrix().getType()));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("symmetric", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct TanOpCastPattern : public ModelicaOpRewritePattern<TanOp>
  {
    using ModelicaOpRewritePattern<TanOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(TanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(TanOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<TanOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct TanOpLowering : public ModelicaOpConversionPattern<TanOp>
  {
    using ModelicaOpConversionPattern<TanOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(TanOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(TanOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("tan", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct TanhOpCastPattern : public ModelicaOpRewritePattern<TanhOp>
  {
    using ModelicaOpRewritePattern<TanhOp>::ModelicaOpRewritePattern;

    mlir::LogicalResult match(TanhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          !operandType.isa<RealType>() || !resultType.isa<RealType>());
    }

    void rewrite(TanhOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto realType = RealType::get(op.getContext());
      mlir::Value operand = rewriter.create<CastOp>(loc, realType, op.getOperand());
      mlir::Value result = rewriter.create<TanhOp>(loc, realType, operand);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
    }
  };

  struct TanhOpLowering : public ModelicaOpConversionPattern<TanhOp>
  {
    using ModelicaOpConversionPattern<TanhOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult match(TanhOp op) const override
    {
      mlir::Type operandType = op.getOperand().getType();
      mlir::Type resultType = op.getResult().getType();

      return mlir::LogicalResult::success(
          operandType.isa<RealType>() && resultType.isa<RealType>());
    }

    void rewrite(TanhOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      assert(op.getOperand().getType().isa<RealType>());
      newOperands.push_back(adaptor.getOperand());
      mangledArgsTypes.push_back(getMangledType(op.getOperand().getType()));

      // Create the call to the runtime library
      assert(op.getResult().getType().isa<RealType>());
      auto resultType = getTypeConverter()->convertType(op.getResult().getType());
      auto mangledResultType = getMangledType(op.getResult().getType());

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("tanh", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
    }
  };

  struct TransposeOpLowering : public ModelicaOpConversionPattern<TransposeOp>
  {
    using ModelicaOpConversionPattern<TransposeOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 2> newOperands;
      llvm::SmallVector<std::string, 2> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      if (!resultType.hasStaticShape()) {
        mlir::Value sourceMatrix = getTypeConverter()->materializeSourceConversion(
            rewriter, op.getMatrix().getLoc(), op.getMatrix().getType(), adaptor.getMatrix());

        if (resultType.getShape()[0] == ArrayType::kDynamicSize) {
          mlir::Value one = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
          dynamicDimensions.push_back(rewriter.create<DimOp>(loc, sourceMatrix, one));
        }

        if (resultType.getShape()[1] == ArrayType::kDynamicSize) {
          mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
          dynamicDimensions.push_back(rewriter.create<DimOp>(loc, sourceMatrix, zero));
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      result = rewriter.create<ArrayCastOp>(loc, UnrankedArrayType::get(resultType.getElementType(), resultType.getMemorySpace()), result);
      newOperands.push_back(materializeTargetConversion(rewriter, result));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Matrix
      assert(op.getMatrix().getType().isa<ArrayType>());

      mlir::Value matrix = adaptor.getMatrix();
      matrix = getTypeConverter()->materializeSourceConversion(rewriter, loc, op.getMatrix().getType(), matrix);
      matrix = rewriter.create<ArrayCastOp>(loc, UnrankedArrayType::get(resultType.getElementType(), resultType.getMemorySpace()), matrix);
      matrix = materializeTargetConversion(rewriter, matrix);
      newOperands.push_back(matrix);

      mangledArgsTypes.push_back(getMangledType(op.getMatrix().getType()));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("transpose", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
      return mlir::success();
    }
  };

  struct ZerosOpLowering : public ModelicaOpConversionPattern<ZerosOp>
  {
    using ModelicaOpConversionPattern<ZerosOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ZerosOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Result
      auto resultType = op.getResult().getType().cast<ArrayType>();
      assert(resultType.getRank() == 2);
      llvm::SmallVector<mlir::Value, 3> dynamicDimensions;

      for (const auto& size : resultType.getShape()) {
        if (size == ArrayType::kDynamicSize) {
          auto index = dynamicDimensions.size();
          mlir::Value dimensionSize = adaptor.getSizes()[index];

          dimensionSize = getTypeConverter()->materializeSourceConversion(
              rewriter, dimensionSize.getLoc(), op.getSizes()[index].getType(), adaptor.getSizes()[index]);

          dimensionSize = rewriter.create<CastOp>(dimensionSize.getLoc(), rewriter.getIndexType(), dimensionSize);
          dynamicDimensions.push_back(dimensionSize);
        }
      }

      mlir::Value result = rewriter.replaceOpWithNewOp<AllocOp>(op, resultType, dynamicDimensions);
      newOperands.push_back(convertToUnsizedArray(rewriter, materializeTargetConversion(rewriter, result)));
      mangledArgsTypes.push_back(getMangledType(resultType));

      // Create the call to the runtime library
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("zeros", mangledResultType, mangledArgsTypes),
          getVoidType(), newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
      return mlir::success();
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

//===----------------------------------------------------------------------===//
// Utility operations
//===----------------------------------------------------------------------===//

namespace
{
  struct ArrayFillOpLowering : public ModelicaOpConversionPattern<ArrayFillOp>
  {
    using ModelicaOpConversionPattern<ArrayFillOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ArrayFillOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Array
      newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getArray()));
      mangledArgsTypes.push_back(getMangledType(op.getArray().getType()));

      // Value
      newOperands.push_back(adaptor.getValue());
      mangledArgsTypes.push_back(getMangledType(op.getValue().getType()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("fill", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, callee, newOperands);
      return mlir::success();
    }
  };

  struct PrintOpLowering : public ModelicaOpConversionPattern<PrintOp>
  {
    using ModelicaOpConversionPattern<PrintOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(PrintOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      llvm::SmallVector<mlir::Value, 1> newOperands;
      llvm::SmallVector<std::string, 1> mangledArgsTypes;

      // Operand
      if (op.getValue().getType().isa<ArrayType>()) {
        newOperands.push_back(convertToUnsizedArray(rewriter, adaptor.getValue()));
      } else {
        newOperands.push_back(adaptor.getValue());
      }

      mangledArgsTypes.push_back(getMangledType(op.getValue().getType()));

      // Create the call to the runtime library
      auto resultType = getVoidType();
      auto mangledResultType = getMangler()->getVoidType();

      auto callee = getOrDeclareFunction(
          rewriter,
          op->getParentOfType<mlir::ModuleOp>(),
          getMangler()->getMangledFunction("print", mangledResultType, mangledArgsTypes),
          resultType, newOperands);

      rewriter.create<mlir::LLVM::CallOp>(loc, callee, newOperands);
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

  // Math operations
  patterns.insert<
      PowOpLowering>(context, typeConverter, options);

  // Built-in functions
  patterns.insert<
      AbsOpCastPattern,
      AcosOpCastPattern,
      AsinOpCastPattern,
      AtanOpCastPattern,
      Atan2OpCastPattern,
      CeilOpCastPattern,
      CosOpCastPattern,
      CoshOpCastPattern,
      DivTruncOpCastPattern,
      ExpOpCastPattern,
      FloorOpCastPattern,
      IdentityOpCastPattern,
      IntegerOpCastPattern,
      LinspaceOpCastPattern,
      LogOpCastPattern,
      Log10OpCastPattern,
      MaxOpArrayCastPattern,
      MaxOpScalarsCastPattern,
      MinOpArrayCastPattern,
      MinOpScalarsCastPattern,
      ModOpCastPattern,
      NDimsOpLowering,
      ProductOpCastPattern,
      RemOpCastPattern,
      SignOpCastPattern,
      SinOpCastPattern,
      SinhOpCastPattern,
      SizeOpDimensionLowering,
      SizeOpArrayLowering,
      SqrtOpCastPattern,
      SumOpCastPattern,
      TanOpCastPattern,
      TanhOpCastPattern>(context, options);

  patterns.insert<
      AbsOpLowering,
      AcosOpLowering,
      AsinOpLowering,
      AtanOpLowering,
      Atan2OpLowering,
      CeilOpLowering,
      CosOpLowering,
      CoshOpLowering,
      DiagonalOpLowering,
      DivTruncOpLowering,
      ExpOpLowering,
      FloorOpLowering,
      IdentityOpLowering,
      IntegerOpLowering,
      LinspaceOpLowering,
      LogOpLowering,
      Log10OpLowering,
      OnesOpLowering,
      MaxOpArrayLowering,
      MaxOpScalarsLowering,
      MinOpArrayLowering,
      MinOpScalarsLowering,
      ModOpLowering,
      ProductOpLowering,
      RemOpLowering,
      SignOpLowering,
      SignOpLowering,
      SinOpLowering,
      SinhOpLowering,
      SqrtOpLowering,
      SumOpLowering,
      SymmetricOpLowering,
      TanOpLowering,
      TanhOpLowering,
      TransposeOpLowering,
      ZerosOpLowering>(context, typeConverter, options);

  // Various operations
  patterns.insert<
      AssignmentOpScalarLowering,
      AssignmentOpArrayLowering>(context, options);

  // Utility operations
  patterns.insert<
      ArrayFillOpLowering,
      PrintOpLowering>(context, typeConverter, options);
}

#include "llvm/Support/Debug.h"

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
      }

    private:
      mlir::LogicalResult convertOperations()
      {
        auto module = getOperation();
        mlir::ConversionTarget target(getContext());

        target.addIllegalOp<CastOp>();

        target.addDynamicallyLegalOp<PowOp>([](PowOp op) {
          return !isNumeric(op.getBase());
        });

        target.addIllegalOp<
            AbsOp,
            AcosOp,
            AsinOp,
            AtanOp,
            Atan2Op,
            CeilOp,
            CosOp,
            CoshOp,
            DiagonalOp,
            DivTruncOp,
            ExpOp,
            FloorOp,
            IdentityOp,
            IntegerOp,
            LinspaceOp,
            LogOp,
            Log10Op,
            OnesOp,
            MaxOp,
            MinOp,
            ModOp,
            NDimsOp,
            ProductOp,
            RemOp,
            SignOp,
            SinOp,
            SinhOp,
            SizeOp,
            SqrtOp,
            SumOp,
            SymmetricOp,
            TanOp,
            TanhOp,
            TransposeOp,
            ZerosOp>();

        target.addIllegalOp<
            AssignmentOp>();

        target.addIllegalOp<
            ArrayFillOp,
            PrintOp>();

        // Some operations, once converted, do create operations operating on
        // arrays, such as allocations or copies.
        //target.addIllegalOp<AllocOp, AllocaOp, LoadOp, FreeOp, DimOp>();

        //target.addIllegalDialect<mlir::memref::MemRefDialect>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        llvmLoweringOptions.dataLayout = options.dataLayout;
        mlir::modelica::LLVMTypeConverter llvmTypeConverter(&getContext(), llvmLoweringOptions, options.bitWidth);

        mlir::RewritePatternSet patterns(&getContext());

        // TODO PROVARE A SETTARE UN TYPE CONVERTER CHE CONVERTE ARRAY IN MEMREF.
        // DEVE COMUNQUE ESSERE UN LLVMTYPECONVERTER. POI NEL PATTERN FACCIO CAST A UNRANKEDMEMREF, E POI TARGET MATERIALIZATION

        populateModelicaToLLVMPatterns(patterns, &getContext(), llvmTypeConverter, options);

        /*
        ModelicaToMemRefOptions modelicaToMemRefOptions;
        modelicaToMemRefOptions.bitWidth = options.bitWidth;
        modelicaToMemRefOptions.assertions = options.assertions;
        modelicaToMemRefOptions.dataLayout = options.dataLayout;

        mlir::modelica::TypeConverter baseTypeConverter(options.bitWidth);
        */
        //populateModelicaToMemRefPatterns(patterns, &getContext(), baseTypeConverter, modelicaToMemRefOptions);

        //mlir::populateMemRefToLLVMConversionPatterns(llvmTypeConverter, patterns);

        //populateIDAStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

        llvm::DebugFlag = true;

        mlir::OpBuilder builder(&getContext());
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
