#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "marco/Codegen/Conversion/PassDetail.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static bool isNumeric(mlir::Type type)
{
  return type.isa<BooleanType, IntegerType, RealType, mlir::IndexType>();
}

static bool isNumeric(mlir::Value value)
{
  return isNumeric(value.getType());
}

static std::vector<mlir::Value> getArrayDynamicDimensions(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value array)
{
  std::vector<mlir::Value> result;

  assert(array.getType().isa<ArrayType>());
  auto arrayType = array.getType().cast<ArrayType>();

  for (const auto& dimension : llvm::enumerate(arrayType.getShape())) {
    if (dimension.value() == ArrayType::kDynamicSize) {
      mlir::Value dim = builder.create<ConstantOp>(loc, builder.getIndexAttr(dimension.index()));
      result.push_back(builder.create<DimOp>(loc, array, dim));
    }
  }

  return result;
}

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

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::Type result, llvm::ArrayRef<mlir::Type> args)
{
	if (auto funcOp = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
    return funcOp;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  return builder.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), name, mlir::LLVM::LLVMFunctionType::get(result, args));
}

static mlir::LLVM::LLVMFuncOp getOrDeclareFunction(mlir::OpBuilder& builder, mlir::ModuleOp module, llvm::StringRef name, mlir::Type result, mlir::ValueRange args)
{
  llvm::SmallVector<mlir::Type, 3> argsTypes;

  for (const auto& arg : args) {
    argsTypes.push_back(arg.getType());
  }

	return getOrDeclareFunction(builder, module, name, result, argsTypes);
}

static mlir::Type castToMostGenericType(
    mlir::OpBuilder& builder, mlir::ValueRange values, llvm::SmallVectorImpl<mlir::Value>& castedValues)
{
	mlir::Type resultType = nullptr;
	mlir::Type resultBaseType = nullptr;

	for (const auto& value : values) {
		mlir::Type type = value.getType();
		mlir::Type baseType = type;

		if (resultType == nullptr) {
			resultType = type;
			resultBaseType = type;

			while (resultBaseType.isa<ArrayType>()) {
        resultBaseType = resultBaseType.cast<ArrayType>().getElementType();
      }

			continue;
		}

		if (type.isa<ArrayType>()) {
			while (baseType.isa<ArrayType>()) {
        baseType = baseType.cast<ArrayType>().getElementType();
      }
		}

		if (resultBaseType.isa<mlir::IndexType>() || baseType.isa<RealType>()) {
			resultType = type;
			resultBaseType = baseType;
		}
	}

	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& value : values)
	{
		mlir::Type type = value.getType();

		if (type.isa<ArrayType>())
		{
			auto arrayType = type.cast<ArrayType>();
			auto shape = arrayType.getShape();
			types.emplace_back(ArrayType::get(arrayType.getContext(), resultBaseType, shape));
		}
		else
			types.emplace_back(resultBaseType);
	}

	for (const auto& [value, type] : llvm::zip(values, types))
	{
		mlir::Value castedValue = builder.create<CastOp>(value.getLoc(), type, value);
		castedValues.push_back(castedValue);
	}

	return types[0];
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

  struct ArrayCastOpLowering : public ModelicaOpConversionPattern<ArrayCastOp>
  {
    using ModelicaOpConversionPattern<ArrayCastOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(ArrayCastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      mlir::Type sourceType = op.getSource().getType();
      auto resultType = op.getResult().getType();

      if (!sourceType.isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an array");
      }

      auto sourceArrayType = sourceType.cast<ArrayType>();
      ArrayDescriptor sourceDescriptor(this->getTypeConverter(), adaptor.getSource());

      if (!resultType.isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an array");
      }

      auto resultArrayType = resultType.cast<ArrayType>();

      if (sourceArrayType.getRank() != resultArrayType.getRank()) {
        return rewriter.notifyMatchFailure(op, "The destination array type has a different rank");
      }

      for (const auto& dimension : resultArrayType.getShape()) {
        if (dimension != ArrayType::kDynamicSize) {
          return rewriter.notifyMatchFailure(op, "The destination array type has some fixed dimensions");
        }
      }

      ArrayDescriptor resultDescriptor = ArrayDescriptor::undef(
          rewriter, getTypeConverter(), loc, getTypeConverter()->convertType(resultType));

      resultDescriptor.setPtr(rewriter, loc, sourceDescriptor.getPtr(rewriter, loc));
      resultDescriptor.setRank(rewriter, loc, sourceDescriptor.getRank(rewriter, loc));

      for (unsigned int i = 0; i < sourceArrayType.getRank(); ++i) {
        resultDescriptor.setSize(rewriter, loc, i, sourceDescriptor.getSize(rewriter, loc, i));
      }

      mlir::Value result = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, *resultDescriptor);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };
}

//===----------------------------------------------------------------------===//
// Array operations
//===----------------------------------------------------------------------===//

namespace
{
  template<typename FromOp>
  struct AllocLikeOpLowering : public ModelicaOpConversionPattern<FromOp>
  {
    public:
    using ModelicaOpConversionPattern<FromOp>::ModelicaOpConversionPattern;
    using OpAdaptor = typename ModelicaOpConversionPattern<FromOp>::OpAdaptor;

    protected:
    virtual ArrayType getResultType(FromOp op) const = 0;
    virtual mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, FromOp op, mlir::Value sizeBytes) const = 0;

    public:
    mlir::LogicalResult matchAndRewrite(FromOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto arrayType = getResultType(op);
      mlir::Type indexType = this->getTypeConverter()->convertType(rewriter.getIndexType());

      // Create the descriptor
      auto descriptor = ArrayDescriptor::undef(
          rewriter, this->getTypeConverter(), loc,
          this->getTypeConverter()->convertType(arrayType));

      mlir::Type sizeType = descriptor.getSizeType();

      // Save the rank into the descriptor
      mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, descriptor.getRankType(), rewriter.getIntegerAttr(descriptor.getRankType(), arrayType.getRank()));

      descriptor.setRank(rewriter, loc, rank);

      // Determine the total size of the array in bytes
      auto shape = arrayType.getShape();
      llvm::SmallVector<mlir::Value, 3> sizes;

      // Multi-dimensional arrays must be flattened into a 1-dimensional one.
      // For example, v[s1][s2][s3] becomes v[s1 * s2 * s3] and the access rule
      // is such that v[i][j][k] = v[(i * s1 + j) * s2 + k].

      mlir::Value totalSize = rewriter.create<mlir::LLVM::ConstantOp>(loc, sizeType, rewriter.getIntegerAttr(sizeType, 1));

      for (size_t i = 0, dynamicDimensions = 0, end = shape.size(); i < end; ++i) {
        long dimension = shape[i];

        if (dimension == ArrayType::kDynamicSize) {
          mlir::Value size = adaptor.getOperands()[dynamicDimensions++];
          sizes.push_back(size);
        } else {
          mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(loc, sizeType, rewriter.getIntegerAttr(sizeType, dimension));
          sizes.push_back(size);
        }

        totalSize = rewriter.create<mlir::LLVM::MulOp>(loc, sizeType, totalSize, sizes[i]);
      }

      // Determine the buffer size in bytes
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(
          this->getTypeConverter()->convertType(arrayType.getElementType()));

      mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
      mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, totalSize);
      mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, gepPtr);

      // Allocate the underlying buffer and store the pointer into the descriptor
      mlir::Value buffer = allocateBuffer(rewriter, loc, op, sizeBytes);
      descriptor.setPtr(rewriter, loc, buffer);

      // Store the sizes into the descriptor
      for (auto size : llvm::enumerate(sizes)) {
        descriptor.setSize(rewriter, loc, size.index(), size.value());
      }

      rewriter.replaceOp(op, *descriptor);
      return mlir::success();
    }
  };

  class AllocaOpLowering : public AllocLikeOpLowering<AllocaOp>
  {
    using AllocLikeOpLowering<AllocaOp>::AllocLikeOpLowering;

    ArrayType getResultType(AllocaOp op) const override
    {
      return op.getArrayType();
    }

    mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocaOp op, mlir::Value sizeBytes) const override
    {
      mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->convertType(op.getArrayType().getElementType()));

      return rewriter.create<mlir::LLVM::AllocaOp>(loc, bufferPtrType, sizeBytes, op->getAttrs());
    }
  };

  class AllocOpLowering : public AllocLikeOpLowering<AllocOp>
  {
    using AllocLikeOpLowering<AllocOp>::AllocLikeOpLowering;

    ArrayType getResultType(AllocOp op) const override
    {
      return op.getArrayType();
    }

    mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocOp op, mlir::Value sizeBytes) const override
    {
      // Insert the "malloc" declaration if it is not already present in the module
      auto heapAllocFunc = lookupOrCreateHeapAllocFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

      // Allocate the buffer
      mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(
          getTypeConverter()->convertType(op.getArrayType().getElementType()));

      auto results = createLLVMCall(rewriter, loc, heapAllocFunc, sizeBytes, getVoidPtrType());
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, bufferPtrType, results[0]);
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapAlloc_pvoid_i64";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(), builder.getI64Type());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }
  };

  class FreeOpLowering : public ModelicaOpConversionPattern<FreeOp>
  {
    using ModelicaOpConversionPattern<FreeOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(FreeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Insert the "free" declaration if it is not already present in the module
      auto freeFunc = lookupOrCreateHeapFreeFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

      // Extract the buffer address and call the "free" function
      ArrayDescriptor descriptor(getTypeConverter(), adaptor.getArray());
      mlir::Value address = descriptor.getPtr(rewriter, loc);
      mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), address);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, freeFunc, casted);

      return mlir::success();
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapFree_void_pvoid";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidType(), getVoidPtrType());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }
  };

  class DimOpLowering : public ModelicaOpConversionPattern<DimOp>
  {
    using ModelicaOpConversionPattern<DimOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(DimOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // The actual size of each dimension is stored in the memory description
      // structure.
      ArrayDescriptor descriptor(this->getTypeConverter(), adaptor.getArray());
      mlir::Value size = descriptor.getSize(rewriter, loc, adaptor.getDimension());

      rewriter.replaceOp(op, size);
      return mlir::success();
    }
  };

  class SubscriptOpLowering : public ModelicaOpConversionPattern<SubscriptionOp>
  {
    using ModelicaOpConversionPattern<SubscriptionOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(SubscriptionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      auto sourceArrayType = op.getSourceArrayType();
      auto resultArrayType = op.getResultArrayType();

      ArrayDescriptor sourceDescriptor(getTypeConverter(), adaptor.getSource());

      ArrayDescriptor result = ArrayDescriptor::undef(
          rewriter, getTypeConverter(), loc,
          getTypeConverter()->convertType(resultArrayType));

      mlir::Value index = adaptor.getIndices()[0];

      for (size_t i = 1, e = sourceArrayType.getRank(); i < e; ++i) {
        mlir::Value size = sourceDescriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, getTypeConverter()->getIndexType(), index, size);

        if (i < adaptor.getIndices().size()) {
          mlir::Value offset = adaptor.getIndices()[i];
          index = rewriter.create<mlir::LLVM::AddOp>(loc, getTypeConverter()->getIndexType(), index, offset);
        }
      }

      mlir::Value base = sourceDescriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);
      result.setPtr(rewriter, loc, ptr);

      mlir::Type rankType = result.getRankType();
      mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, rankType, rewriter.getIntegerAttr(rankType, resultArrayType.getRank()));
      result.setRank(rewriter, loc, rank);

      for (size_t i = sourceArrayType.getRank() - resultArrayType.getRank(), e = sourceArrayType.getRank(), j = 0; i < e; ++i, ++j) {
        result.setSize(rewriter, loc, j, sourceDescriptor.getSize(rewriter, loc, i));
      }

      rewriter.replaceOp(op, *result);
      return mlir::success();
    }
  };

  class LoadOpLowering : public ModelicaOpConversionPattern<LoadOp>
  {
    using ModelicaOpConversionPattern<LoadOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto indexes = adaptor.getIndices();

      assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

      // Determine the address into which the value has to be stored.
      ArrayDescriptor descriptor(getTypeConverter(), adaptor.getArray());
      auto indexType = getTypeConverter()->convertType(rewriter.getIndexType());

      auto indexFn = [&]() -> mlir::Value {
        if (indexes.empty()) {
          return rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));
        }

        return indexes[0];
      };

      mlir::Value index = indexFn();

      for (size_t i = 1, e = indexes.size(); i < e; ++i) {
        mlir::Value size = descriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);
        index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, indexes[i]);
      }

      mlir::Value base = descriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

      // Load the value
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptr);

      return mlir::success();
    }
  };

  class StoreOpLowering : public ModelicaOpConversionPattern<StoreOp>
  {
    using ModelicaOpConversionPattern<StoreOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto indexes = adaptor.getIndices();

      assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

      // Determine the address into which the value has to be stored.
      ArrayDescriptor memoryDescriptor(this->getTypeConverter(), adaptor.getArray());

      mlir::Value index = indexes.empty()
          ? rewriter.create<mlir::LLVM::ConstantOp>(loc, getTypeConverter()->getIndexType(), rewriter.getIndexAttr(0))
          : indexes[0];

      for (size_t i = 1, e = indexes.size(); i < e; ++i) {
        mlir::Value size = memoryDescriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, getTypeConverter()->getIndexType(), index, size);
        index = rewriter.create<mlir::LLVM::AddOp>(loc, getTypeConverter()->getIndexType(), index, indexes[i]);
      }

      mlir::Value base = memoryDescriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

      // Store the value
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(), ptr);

      return mlir::success();
    }
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OEQ, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::ONE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGT, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OGE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLT, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
        return mlir::success();

      } else if (operandsType.isa<mlir::FloatType>()) {
        mlir::Value result = rewriter.create<mlir::arith::CmpFOp>(
            loc, rewriter.getIntegerType(1), mlir::arith::CmpFPredicate::OLE, adaptor.getLhs(), adaptor.getRhs());

        result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(result.getContext()), result);
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);
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
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

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
            mlir::Value dimensionIndex = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(i));
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

  struct AndOpLowering : public ModelicaOpConversionPattern<AndOp>
  {
    using ModelicaOpConversionPattern<AndOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(AndOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a Boolean");
      }

      if (!op.getRhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a Boolean");
      }

      // Compute the result
      mlir::Value result = rewriter.create<mlir::arith::AndIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

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

    mlir::Value scalarize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elementType, mlir::Value lhs, mlir::Value rhs) const override
    {
      return builder.create<AndOp>(loc, elementType, lhs, rhs);
    }
  };

  struct OrOpLowering : public ModelicaOpConversionPattern<OrOp>
  {
    using ModelicaOpConversionPattern<OrOp>::ModelicaOpConversionPattern;

    mlir::LogicalResult matchAndRewrite(OrOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // Check if the operands are compatible
      if (!op.getLhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Left-hand side operand is not a Boolean");
      }

      if (!op.getRhs().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Right-hand side operand is not a Boolean");
      }

      // Compute the result
      mlir::Value result = rewriter.create<mlir::arith::OrIOp>(loc, adaptor.getLhs(), adaptor.getRhs());
      result = getTypeConverter()->materializeSourceConversion(rewriter, loc, BooleanType::get(op->getContext()), result);
      rewriter.replaceOpWithNewOp<CastOp>(op, op.getResult().getType(), result);

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

      // Check if the operand is compatible
      if (!isNumeric(op.getOperand())) {
        return rewriter.notifyMatchFailure(op, "Left-hand side value is not a scalar");
      }

      mlir::Type type = op.getOperand().getType();

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

      mlir::Value lhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getLhs().getLoc(), op.getLhs().getType(), adaptor.getLhs());
      mlir::Value rhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getRhs().getLoc(), op.getRhs().getType(), adaptor.getRhs());

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

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultType));

      init = getTypeConverter()->materializeTargetConversion(
          rewriter, init.getLoc(), getTypeConverter()->convertType(init.getType()), init);

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

      mlir::Value lhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getLhs().getLoc(), op.getLhs().getType(), adaptor.getLhs());
      mlir::Value rhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getRhs().getLoc(), op.getRhs().getType(), adaptor.getRhs());

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

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      init = getTypeConverter()->materializeTargetConversion(
          rewriter, init.getLoc(), getTypeConverter()->convertType(init.getType()), init);

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

      mlir::Value lhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getLhs().getLoc(), op.getLhs().getType(), adaptor.getLhs());
      mlir::Value rhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getRhs().getLoc(), op.getRhs().getType(), adaptor.getRhs());

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

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      init = getTypeConverter()->materializeTargetConversion(
          rewriter, init.getLoc(), getTypeConverter()->convertType(init.getType()), init);

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

      mlir::Value lhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getLhs().getLoc(), op.getLhs().getType(), adaptor.getLhs());
      mlir::Value rhs = getTypeConverter()->materializeSourceConversion(rewriter, op.getRhs().getLoc(), op.getRhs().getType(), adaptor.getRhs());

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

      mlir::Value init = rewriter.create<ConstantOp>(loc, getZeroAttr(resultArrayType.getElementType()));

      init = getTypeConverter()->materializeTargetConversion(
          rewriter, init.getLoc(), getTypeConverter()->convertType(init.getType()), init);

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

      productResult = getTypeConverter()->materializeTargetConversion(
          rewriter, productResult.getLoc(), getTypeConverter()->convertType(productResult.getType()), productResult);

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
            mlir::Value zeroValue = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));

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
      mlir::Value result = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(arrayType.getRank()));
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
        if (!op.getMatrix().getType().cast<ArrayType>().hasConstantShape()) {
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

      if (!resultType.hasConstantShape()) {
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

      if (!resultType.hasConstantShape()) {
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
		mlir::modelica::TypeConverter& typeConverter,
		ModelicaToLLVMOptions options)
{
  // Cast operations
  patterns.insert<
      CastOpIntegerLowering,
      CastOpFloatLowering,
      ArrayCastOpLowering>(context, typeConverter, options);

  // Array operations
  patterns.insert<
      AllocaOpLowering,
      AllocOpLowering,
      FreeOpLowering,
      DimOpLowering,
      SubscriptOpLowering,
      LoadOpLowering,
      StoreOpLowering>(context, typeConverter, options);

  // Logic operations
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
      NotOpLowering,
      AndOpLowering,
      OrOpLowering>(context, typeConverter, options);

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
      DivOpFloatLowering,
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

  patterns.insert<
      SelectOpCastPattern>(context, options);

  patterns.insert<
      SelectOpLowering>(context, typeConverter, options);

  // Utility operations
  patterns.insert<
      ArrayFillOpLowering,
      PrintOpLowering>(context, typeConverter, options);
}

namespace
{
  class ModelicaToLLVMConversionPass : public ModelicaToLLVMBase<ModelicaToLLVMConversionPass>
  {
    public:
      ModelicaToLLVMConversionPass(ModelicaToLLVMOptions options, unsigned int bitWidth)
          : options(std::move(options)),
            bitWidth(bitWidth)
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

        target.addIllegalDialect<ModelicaDialect>();

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
        llvmLoweringOptions.dataLayout = options.dataLayout;
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToLLVMPatterns(patterns, &getContext(), typeConverter, options);
        populateIDAStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

        return applyPartialConversion(module, target, std::move(patterns));
      }

      private:
        ModelicaToLLVMOptions options;
        unsigned int bitWidth;
  };
}

namespace marco::codegen
{
  const ModelicaToLLVMOptions& ModelicaToLLVMOptions::getDefaultOptions()
  {
    static ModelicaToLLVMOptions options;
    return options;
  }

  std::unique_ptr<mlir::Pass> createModelicaToLLVMPass(ModelicaToLLVMOptions options, unsigned int bitWidth)
  {
    return std::make_unique<ModelicaToLLVMConversionPass>(options, bitWidth);
  }
}
