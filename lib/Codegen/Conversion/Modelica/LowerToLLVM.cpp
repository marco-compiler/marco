#include "marco/Codegen/Conversion/Modelica/LowerToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/ArrayDescriptor.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Support/MathExtras.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

/// Generic conversion pattern that provides some utility functions.
template<typename FromOp>
class ModelicaOpConversion : public mlir::ConvertOpToLLVMPattern<FromOp>
{
	protected:
	using Adaptor = typename FromOp::Adaptor;
	using mlir::ConvertOpToLLVMPattern<FromOp>::ConvertOpToLLVMPattern;

	public:
	mlir::Type convertType(mlir::Type type) const
	{
		return this->getTypeConverter()->convertType(type);
	}

	mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
	{
		mlir::Type type = this->getTypeConverter()->convertType(value.getType());
		return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
	}
};

template<typename FromOp>
struct AllocLikeOpLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	protected:
	virtual ArrayType getResultType(FromOp op) const = 0;
	virtual mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, FromOp op, mlir::Value sizeBytes) const = 0;

	public:
	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		auto arrayType = getResultType(op);
		auto typeConverter = this->getTypeConverter();
		mlir::Type indexType = this->convertType(rewriter.getIndexType());

		// Create the descriptor
		auto descriptor = ArrayDescriptor::undef(rewriter, typeConverter, loc, this->convertType(arrayType));
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

		for (size_t i = 0, dynamicDimensions = 0, end = shape.size(); i < end; ++i)
		{
			long dimension = shape[i];

			if (dimension == -1)
			{
				mlir::Value size = operands[dynamicDimensions++];
				sizes.push_back(size);
			}
			else
			{
				mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(loc, sizeType, rewriter.getIntegerAttr(sizeType, dimension));
				sizes.push_back(size);
			}

			totalSize = rewriter.create<mlir::LLVM::MulOp>(loc, sizeType, totalSize, sizes[i]);
		}

		// Determine the buffer size in bytes
		mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(this->convertType(arrayType.getElementType()));
		mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
		mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, llvm::ArrayRef<mlir::Value>{ nullPtr, totalSize });
		mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, gepPtr);

		// Allocate the underlying buffer and store the pointer into the descriptor
		mlir::Value buffer = allocateBuffer(rewriter, loc, op, sizeBytes);
		descriptor.setPtr(rewriter, loc, buffer);

		// Store the sizes into the descriptor
		for (auto size : llvm::enumerate(sizes))
			descriptor.setSize(rewriter, loc, size.index(), size.value());

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
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getArrayType().getElementType()));
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
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getArrayType().getElementType()));
		auto results = createLLVMCall(rewriter, loc, heapAllocFunc, sizeBytes, getVoidPtrType());
		return rewriter.create<mlir::LLVM::BitcastOp>(loc, bufferPtrType, results[0]);
	}

	mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
	{
		std::string name = "_MheapAlloc_pvoid_i64";

		if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
			return foo;

		mlir::PatternRewriter::InsertionGuard insertGuard(builder);
		builder.setInsertionPointToStart(module.getBody());
		auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(), builder.getI64Type());
		return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
	}
};

class FreeOpLowering: public ModelicaOpConversion<FreeOp>
{
	using ModelicaOpConversion<FreeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FreeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor adaptor(operands);
		auto typeConverter = this->getTypeConverter();

		// Insert the "free" declaration if it is not already present in the module
		auto freeFunc = lookupOrCreateHeapFreeFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

		// Extract the buffer address and call the "free" function
		ArrayDescriptor descriptor(typeConverter, adaptor.array());
		mlir::Value address = descriptor.getPtr(rewriter, loc);
		mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), address);
		rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, llvm::None, rewriter.getSymbolRefAttr(freeFunc), casted);

		return mlir::success();
	}

	mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
	{
		std::string name = "_MheapFree_void_pvoid";

		if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
			return foo;

		mlir::PatternRewriter::InsertionGuard insertGuard(builder);
		builder.setInsertionPointToStart(module.getBody());
		auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidType(), getVoidPtrType());
		return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
	}
};

class DimOpLowering: public ModelicaOpConversion<DimOp>
{
	using ModelicaOpConversion<DimOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(DimOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		// The actual size of each dimensions is stored in the memory description
		// structure.
		ArrayDescriptor descriptor(this->getTypeConverter(), adaptor.array());
		mlir::Value size = descriptor.getSize(rewriter, location, adaptor.dimension());

		rewriter.replaceOp(op, size);
		return mlir::success();
	}
};

class SubscriptOpLowering : public ModelicaOpConversion<SubscriptionOp>
{
	using ModelicaOpConversion<SubscriptionOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SubscriptionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor adaptor(operands);
		auto typeConverter = this->getTypeConverter();
		mlir::Type indexType = convertType(rewriter.getIndexType());

		auto sourceArrayType = op.getSourceArrayType();
    auto resultArrayType = op.getResultArrayType();

		ArrayDescriptor sourceDescriptor(typeConverter, adaptor.source());
		ArrayDescriptor result = ArrayDescriptor::undef(rewriter, typeConverter, loc, convertType(resultArrayType));

		mlir::Value index = adaptor.indices()[0];

		for (size_t i = 1, e = sourceArrayType.getRank(); i < e; ++i)
		{
			mlir::Value size = sourceDescriptor.getSize(rewriter, loc, i);
			index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);

			if (i < adaptor.indices().size())
			{
				mlir::Value offset = adaptor.indices()[i];
				index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, offset);
			}
		}

		mlir::Value base = sourceDescriptor.getPtr(rewriter, loc);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);
		result.setPtr(rewriter, loc, ptr);

		mlir::Type rankType = result.getRankType();
		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, rankType, rewriter.getIntegerAttr(rankType, resultArrayType.getRank()));
		result.setRank(rewriter, loc, rank);

		for (size_t i = sourceArrayType.getRank() - resultArrayType.getRank(), e = sourceArrayType.getRank(), j = 0; i < e; ++i, ++j)
			result.setSize(rewriter, loc, j, sourceDescriptor.getSize(rewriter, loc, i));

		rewriter.replaceOp(op, *result);
		return mlir::success();
	}
};

class LoadOpLowering: public ModelicaOpConversion<LoadOp>
{
	using ModelicaOpConversion<LoadOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LoadOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		auto typeConverter = this->getTypeConverter();
		mlir::Location loc = op->getLoc();
		Adaptor adaptor(operands);
		auto indexes = adaptor.indices();

			assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		ArrayDescriptor descriptor(typeConverter, adaptor.array());
		auto indexType = convertType(rewriter.getIndexType());

		auto indexFn = [&]() -> mlir::Value {
			if (indexes.empty())
				return rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));

			return indexes[0];
		};

		mlir::Value index = indexFn();

		for (size_t i = 1, e = indexes.size(); i < e; ++i)
		{
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

class StoreOpLowering: public ModelicaOpConversion<StoreOp>
{
	using ModelicaOpConversion<StoreOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(StoreOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor adaptor(operands);
		auto indexes = adaptor.indices();

		assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		ArrayDescriptor memoryDescriptor(this->getTypeConverter(), adaptor.array());

		auto indexType = convertType(rewriter.getIndexType());
		mlir::Value index = indexes.empty() ? rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0)) : indexes[0];

		for (size_t i = 1, e = indexes.size(); i < e; ++i)
		{
			mlir::Value size = memoryDescriptor.getSize(rewriter, loc, i);
			index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);
			index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, indexes[i]);
		}

		mlir::Value base = memoryDescriptor.getPtr(rewriter, loc);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

		// Store the value
		rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), ptr);

		return mlir::success();
	}
};

class CastOpIndexLowering: public ModelicaOpConversion<CastOp>
{
	using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		if (!op.value().getType().isa<mlir::IndexType>())
			return rewriter.notifyMatchFailure(op, "Source is not an IndexType");

		mlir::Location location = op.getLoc();

		auto source = op.value().getType().cast<mlir::IndexType>();
		mlir::Type destination = op.getResult().getType();

		if (source == destination)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		if (destination.isa<IntegerType>())
		{
			rewriter.replaceOpWithNewOp<mlir::IndexCastOp>(op, op.value(), convertType(destination));
			return mlir::success();
		}

		if (destination.isa<RealType>())
		{
			mlir::Value value = rewriter.create<mlir::IndexCastOp>(location, op.value(), convertType(IntegerType::get(rewriter.getContext())));
			value = materializeTargetConversion(rewriter, value);
			rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
	}
};

class CastOpBooleanLowering: public ModelicaOpConversion<CastOp>
{
	using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		if (!op.value().getType().isa<BooleanType>())
			return rewriter.notifyMatchFailure(op, "Source is not a BooleanType");

		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		auto source = op.value().getType().cast<BooleanType>();
		mlir::Type destination = op.getResult().getType();

		if (source == destination)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		if (destination.isa<RealType>())
		{
			mlir::Value value = adaptor.value();
			value = rewriter.create<mlir::LLVM::SExtOp>(location, convertType(IntegerType::get(rewriter.getContext())), value);
			rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
			return mlir::success();
		}

		if (destination.isa<mlir::IndexType>())
		{
			rewriter.replaceOpWithNewOp<mlir::IndexCastOp>(op, adaptor.value(), rewriter.getIndexType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
	}
};

class CastOpIntegerLowering: public ModelicaOpConversion<CastOp>
{
	using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		if (!op.value().getType().isa<IntegerType>())
			return rewriter.notifyMatchFailure(op, "Source is not an IntegerType");

		Adaptor adaptor(operands);

		auto source = op.value().getType().cast<IntegerType>();
		mlir::Type destination = op.getResult().getType();

		if (source == destination)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		if (destination.isa<RealType>())
		{
			mlir::Value value = adaptor.value();
			rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
			return mlir::success();
		}

		if (destination.isa<mlir::IndexType>())
		{
			rewriter.replaceOpWithNewOp<mlir::IndexCastOp>(op, adaptor.value(), rewriter.getIndexType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
	}
};

class CastOpRealLowering: public ModelicaOpConversion<CastOp>
{
	using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		if (!op.value().getType().isa<RealType>())
			return rewriter.notifyMatchFailure(op, "Source is not a RealType");

		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		auto source = op.value().getType().cast<RealType>();
		mlir::Type destination = op.getResult().getType();

		if (source == destination)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		if (destination.isa<IntegerType>())
		{
			mlir::Value value = adaptor.value();
			rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(op, convertType(destination), value);
			return mlir::success();
		}

		if (destination.isa<mlir::IndexType>())
		{
			mlir::Value value = rewriter.create<mlir::LLVM::FPToSIOp>(location, convertType(destination), adaptor.value());
			rewriter.replaceOpWithNewOp<mlir::IndexCastOp>(op, value, rewriter.getIndexType());
			return mlir::success();
		}

		return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
	}
};

struct ArrayCastOpArrayToArrayLowering : public ModelicaOpConversion<ArrayCastOp>
{
  using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

  mlir::LogicalResult matchAndRewrite(ArrayCastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op->getLoc();
    Adaptor transformed(operands);
    mlir::Type sourceType = op.source().getType();
    auto resultType = op.getResult().getType();

    if (!sourceType.isa<ArrayType>()) {
      return rewriter.notifyMatchFailure(op, "Source is not an array");
    }

    auto sourceArrayType = sourceType.cast<ArrayType>();
    ArrayDescriptor sourceDescriptor(this->getTypeConverter(), transformed.source());

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

    ArrayDescriptor resultDescriptor = ArrayDescriptor::undef(rewriter, mlir::ConvertToLLVMPattern::getTypeConverter(), loc, convertType(resultType));
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

struct ArrayCastOpArrayToUnsizedArrayLowering : public ModelicaOpConversion<ArrayCastOp>
{
  using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

  mlir::LogicalResult matchAndRewrite(ArrayCastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op->getLoc();
    Adaptor transformed(operands);
    mlir::Type sourceType = op.source().getType();
    auto resultType = op.getResult().getType();

    if (!sourceType.isa<ArrayType>()) {
      return rewriter.notifyMatchFailure(op, "Source is not an array");
    }

    ArrayDescriptor sourceDescriptor(this->getTypeConverter(), transformed.source());

    if (!resultType.isa<UnsizedArrayType>()) {
      return rewriter.notifyMatchFailure(op, "Result is not an unsized array");
    }

    // Create the unsized array descriptor that holds the ranked one.
    // The inner descriptor (that is, the sized array descriptor) is allocated on the stack.
    UnsizedArrayDescriptor resultDescriptor = UnsizedArrayDescriptor::undef(rewriter, loc, convertType(resultType));
    resultDescriptor.setRank(rewriter, loc, sourceDescriptor.getRank(rewriter, loc));

    mlir::Value underlyingDescPtr = rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), sourceDescriptor.computeSize(rewriter, loc), llvm::None);
    resultDescriptor.setPtr(rewriter, loc, underlyingDescPtr);
    mlir::Type sourceDescriptorArrayType = mlir::LLVM::LLVMPointerType::get(transformed.source().getType());
    underlyingDescPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, sourceDescriptorArrayType, underlyingDescPtr);

    mlir::Type indexType = convertType(rewriter.getIndexType());
    mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));
    mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, underlyingDescPtr.getType(), underlyingDescPtr, zero);
    rewriter.create<mlir::LLVM::StoreOp>(loc, *sourceDescriptor, ptr);

    mlir::Value result = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, *resultDescriptor);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct ArrayCastOpUnsizedArrayToArrayLowering : public ModelicaOpConversion<ArrayCastOp>
{
  using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

  mlir::LogicalResult matchAndRewrite(ArrayCastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
  {
    auto loc = op->getLoc();
    Adaptor transformed(operands);
    mlir::Type sourceType = op.source().getType();
    auto resultType = op.getResult().getType();

    if (!sourceType.isa<UnsizedArrayType>()) {
      return rewriter.notifyMatchFailure(op, "Source is not an unsized array");
    }

    UnsizedArrayDescriptor sourceDescriptor(transformed.source());

    if (!resultType.isa<ArrayType>()) {
      return rewriter.notifyMatchFailure(op, "Result is not an array");
    }

    mlir::Value arrayDescriptorPtr = sourceDescriptor.getPtr(rewriter, loc);

    arrayDescriptorPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc,
        mlir::LLVM::LLVMPointerType::get(getTypeConverter()->convertType(resultType)),
        arrayDescriptorPtr);

    mlir::Value arrayDescriptor = rewriter.create<mlir::LLVM::LoadOp>(loc, arrayDescriptorPtr);
    arrayDescriptor = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, arrayDescriptor);
    rewriter.replaceOp(op, arrayDescriptor);

    return mlir::success();
  }
};

static void populateModelicaToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::OwningRewritePatternList& patterns)
{
	patterns.insert<
			AllocaOpLowering,
			AllocOpLowering,
			FreeOpLowering,
			DimOpLowering,
			SubscriptOpLowering,
			LoadOpLowering,
			StoreOpLowering,
			CastOpIndexLowering,
			CastOpBooleanLowering,
			CastOpIntegerLowering,
			CastOpRealLowering,
      ArrayCastOpArrayToArrayLowering,
      ArrayCastOpArrayToUnsizedArrayLowering,
      ArrayCastOpUnsizedArrayToArrayLowering>(typeConverter);
}

class LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
    explicit LLVMLoweringPass(ModelicaToLLVMConversionOptions options, unsigned int bitWidth)
        : options(std::move(options)), bitWidth(bitWidth)
    {
    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override
    {
      registry.insert<ModelicaDialect>();
      registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() override
    {
      auto module = getOperation();

      if (mlir::failed(stdToLLVMConversionPass(module))) {
        module.emitError("Error in converting to LLVM dialect");
        return signalPassFailure();
      }
    }

	private:
    mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module)
    {
      mlir::LowerToLLVMOptions llvmOptions(&getContext());
      TypeConverter typeConverter(&getContext(), llvmOptions, bitWidth);

      mlir::ConversionTarget target(getContext());
      target.addIllegalDialect<ModelicaDialect, mlir::StandardOpsDialect>();
      target.addIllegalOp<mlir::FuncOp>();

      target.addLegalDialect<mlir::LLVM::LLVMDialect>();
      target.addLegalOp<mlir::UnrealizedConversionCastOp>();
      target.addLegalOp<mlir::ModuleOp>();

      target.addDynamicallyLegalOp<
          mlir::omp::MasterOp,
          mlir::omp::ParallelOp,
          mlir::omp::WsLoopOp>([&](mlir::Operation *op) {
        return typeConverter.isLegal(&op->getRegion(0));
      });

      target.addLegalOp<
          mlir::omp::TerminatorOp,
          mlir::omp::TaskyieldOp,
          mlir::omp::FlushOp,
          mlir::omp::BarrierOp,
          mlir::omp::TaskwaitOp>();

      mlir::OwningRewritePatternList patterns(&getContext());
      populateModelicaToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
      mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

      return applyPartialConversion(module, target, std::move(patterns));
    }

  private:
    ModelicaToLLVMConversionOptions options;
    unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createLLVMLoweringPass(ModelicaToLLVMConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<LLVMLoweringPass>(options, bitWidth);
}
