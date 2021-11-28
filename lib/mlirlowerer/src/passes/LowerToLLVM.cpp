#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Support/MathExtras.h>
#include <marco/mlirlowerer/passes/LowerToLLVM.h>
#include <marco/mlirlowerer/passes/TypeConverter.h>
#include <marco/mlirlowerer/dialects/ida/IdaDialect.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>

using namespace marco::codegen;
using namespace modelica;

/**
 * Helper class to produce LLVM dialect operations extracting or inserting
 * values to a struct representing an array descriptor.
 */
class ArrayDescriptor
{
	public:
	ArrayDescriptor(mlir::LLVMTypeConverter* typeConverter, mlir::Value value)
			: typeConverter(typeConverter),
				value(value),
				descriptorType(value.getType())
	{
		assert(value != nullptr && "Value cannot be null");
		assert(descriptorType.isa<mlir::LLVM::LLVMStructType>() && "Expected LLVM struct type");
	}

	/**
	 * Allocate an empty descriptor.
	 *
	 * @param builder					operation builder
	 * @param location  			source location
	 * @param descriptorType	descriptor type
	 * @return descriptor
	 */
	static ArrayDescriptor undef(mlir::OpBuilder& builder, mlir::LLVMTypeConverter* typeConverter, mlir::Location location, mlir::Type descriptorType)
	{
		mlir::Value descriptor = builder.create<mlir::LLVM::UndefOp>(location, descriptorType);
		return ArrayDescriptor(typeConverter, descriptor);
	}

	[[nodiscard]] mlir::Value operator*()
	{
		return value;
	}

	/**
	 * Build IR to extract the pointer to the memory buffer.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @return memory pointer
	 */
	[[nodiscard]] mlir::Value getPtr(mlir::OpBuilder& builder, mlir::Location location)
	{
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[0];
		return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(0));
	}

	/**
	 * Build IR to set the pointer to the memory buffer.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param ptr       pointer to be set
	 */
	void setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr)
	{
		value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, ptr, builder.getIndexArrayAttr(0));
	}

	/**
	 * Build IR to extract the rank.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @return rank
	 */
	[[nodiscard]] mlir::Value getRank(mlir::OpBuilder& builder, mlir::Location location)
	{
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
		return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(1));
	}

	/**
	 * Build IR to set the rank.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param rank		  rank to be set
	 */
	void setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank)
	{
		value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, rank, builder.getIndexArrayAttr(1));
	}

	/**
	 * Build IR to extract the size of a dimension.
	 *
	 * @param builder    operation builder
	 * @param location   source location
	 * @param dimension  dimension
	 * @return memory pointer
	 */
	[[nodiscard]] mlir::Value getSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension)
	{
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[2];
		type = type.cast<mlir::LLVM::LLVMArrayType>().getElementType();
		return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr({ 2, dimension }));
	}

	/**
	 * Build IR to extract the size of a dimension.
	 *
	 * @param builder    operation builder
	 * @param location   source location
	 * @param dimension  dimension
	 * @return memory pointer
	 */
	[[nodiscard]] mlir::Value getSize(mlir::OpBuilder& builder, mlir::Location location, mlir::Value dimension)
	{
		mlir::Type indexType = typeConverter->convertType(builder.getIndexType());

		mlir::Type sizesContainerType = getSizesContainerType();
		mlir::Value sizes = builder.create<mlir::LLVM::ExtractValueOp>(location, sizesContainerType, value, builder.getIndexArrayAttr(2));

		// Copy size values to stack-allocated memory
		mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(location, indexType, builder.getIntegerAttr(indexType, 1));
		mlir::Value sizesPtr = builder.create<mlir::LLVM::AllocaOp>(location, mlir::LLVM::LLVMPointerType::get(sizesContainerType), one, 0);
		builder.create<mlir::LLVM::StoreOp>(location, sizes, sizesPtr);

		// Load an return size value of interest
		mlir::Type sizeType = getSizeType();
		mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(location, indexType, builder.getIntegerAttr(indexType, 0));
		mlir::Value resultPtr = builder.create<mlir::LLVM::GEPOp>(location, mlir::LLVM::LLVMPointerType::get(sizeType), sizesPtr, mlir::ValueRange({ zero, dimension }));
		return builder.create<mlir::LLVM::LoadOp>(location, resultPtr);
	}

	/**
	 * Build IR to set the size of a dimension.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param ptr       pointer to be set
	 */
	void setSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension, mlir::Value size)
	{
		value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, size, builder.getIndexArrayAttr({ 2, dimension }));
	}

	/**
	 * Emit IR computing the memory necessary to store the descriptor.
	 *
	 * This assumes the descriptor to be
	 *   { type*, i32, i32[rank] }
	 * and densely packed, so the total size is
	 *   sizeof(pointer) + (1 + rank) * sizeof(i32).
	 *
	 * @param builder operation builder
	 * @param loc 	  source location
	 * @return descriptor size in bytes
	 */
	mlir::Value computeSize(mlir::OpBuilder& builder, mlir::Location loc)
	{
		mlir::Type sizeType = getSizeType();
		mlir::Type indexType = typeConverter->convertType(builder.getIndexType());

		mlir::Value pointerSize = builder.create<mlir::LLVM::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, typeConverter->getPointerBitwidth()));

		mlir::Value rank = getRank(builder, loc);
		mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, 1));
		mlir::Value rankIncremented = builder.create<mlir::LLVM::AddOp>(loc, indexType, rank, one);

		mlir::Value integerSize = builder.create<mlir::LLVM::ConstantOp>(loc, indexType, builder.getIntegerAttr(indexType, sizeType.getIntOrFloatBitWidth()));
		mlir::Value rankIntegerSize = builder.create<mlir::LLVM::MulOp>(loc, indexType, rankIncremented, integerSize);

		// Total allocation size
		mlir::Value allocationSize = builder.create<mlir::LLVM::AddOp>(loc, indexType, pointerSize, rankIntegerSize);

		return allocationSize;
	}

	mlir::Type getRankType() const
	{
		auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();
		mlir::Type rankType = body[1];
		assert(rankType.isa<mlir::IntegerType>() && "The rank must have integer type");
		return rankType;
	}

	mlir::Type getSizesContainerType() const
	{
		auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();
		assert(body.size() >= 3);
		auto sizesArrayType = body[2];
		assert(sizesArrayType.isa<mlir::LLVM::LLVMArrayType>() && "The sizes of the array must be contained into an array");
		return sizesArrayType;
	}

	mlir::Type getSizeType() const
	{
		auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();

		if (body.size() == 2)
			return typeConverter->convertType(mlir::IndexType::get(value.getContext()));

		mlir::Type sizesContainerType = getSizesContainerType();
		mlir::Type sizeType = sizesContainerType.cast<mlir::LLVM::LLVMArrayType>().getElementType();
		assert(sizeType.isa<mlir::IntegerType>() && "Each size of the array must have integer type");
		return sizeType;
	}

	private:
	mlir::LLVMTypeConverter* typeConverter;
	mlir::Value value;
	mlir::Type descriptorType;
};

/**
 * Helper class to produce LLVM dialect operations extracting or inserting
 * values to a struct representing an unsized array descriptor.
 */
class UnsizedArrayDescriptor
{
	public:
	explicit UnsizedArrayDescriptor(mlir::Value value)
			: value(value),
				descriptorType(value.getType())
	{
		assert(value != nullptr && "Value cannot be null");
		assert(descriptorType.isa<mlir::LLVM::LLVMStructType>() && "Expected LLVM struct type");
	}

	/**
	 * Allocate an empty descriptor.
	 *
	 * @param builder					operation builder
	 * @param location  			source location
	 * @param descriptorType	descriptor type
	 * @return descriptor
	 */
	static UnsizedArrayDescriptor undef(mlir::OpBuilder& builder, mlir::Location location, mlir::Type descriptorType)
	{
		mlir::Value descriptor = builder.create<mlir::LLVM::UndefOp>(location, descriptorType);
		return UnsizedArrayDescriptor(descriptor);
	}

	[[nodiscard]] mlir::Value operator*()
	{
		return value;
	}

	/**
	 * Build IR to extract the rank.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @return rank
	 */
	[[nodiscard]] mlir::Value getRank(mlir::OpBuilder& builder, mlir::Location location)
	{
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[0];
		return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(0));
	}

	/**
	 * Build IR to set the rank.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param rank		  rank to be set
	 */
	void setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank)
	{
		value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, rank, builder.getIndexArrayAttr(0));
	}

	/**
	 * Build IR to extract the pointer to array descriptor.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @return array descriptor pointer
	 */
	[[nodiscard]] mlir::Value getPtr(mlir::OpBuilder& builder, mlir::Location location)
	{
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
		return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(1));
	}

	/**
	 * Build IR to set the pointer to the array descriptor.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param ptr       pointer to be set
	 */
	void setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr)
	{
		value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, ptr, builder.getIndexArrayAttr(1));
	}

	private:
	mlir::Value value;
	mlir::Type descriptorType;
};

/**
 * Generic conversion pattern that provides some utility functions.
 *
 * @tparam FromOp type of the operation to be converted
 */
template<typename FromOp>
class ModelicaOpConversion : public mlir::ConvertOpToLLVMPattern<FromOp>
{
	protected:
	using Adaptor = typename FromOp::Adaptor;
	using mlir::ConvertOpToLLVMPattern<FromOp>::ConvertOpToLLVMPattern;

	public:
	[[nodiscard]] mlir::Type convertType(mlir::Type type) const
	{
		return this->getTypeConverter()->convertType(type);
	}

	[[nodiscard]] mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
	{
		mlir::Type type = this->getTypeConverter()->convertType(value.getType());
		return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
	}
};

struct PackOpLowering : public ModelicaOpConversion<PackOp>
{
	using ModelicaOpConversion<PackOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PackOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);

		StructType structType = op.resultType();
		mlir::Type descriptorType = convertType(structType);

		mlir::Value result = rewriter.create<mlir::LLVM::UndefOp>(loc, descriptorType);

		for (auto& element : llvm::enumerate(transformed.values()))
			result = rewriter.create<mlir::LLVM::InsertValueOp>(
					loc, descriptorType, result, element.value(), rewriter.getIndexArrayAttr(element.index()));

		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

struct ExtractOpLowering : public ModelicaOpConversion<ExtractOp>
{
	using ModelicaOpConversion<ExtractOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ExtractOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		Adaptor transformed(operands);

		mlir::Type descriptorType = convertType(op.packedValue().getType());
		mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[op.index()];
		mlir::Value result = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, type, transformed.packedValue(), rewriter.getIndexArrayAttr(op.index()));
		rewriter.replaceOp(op, result);
		return mlir::success();
	}
};

template<typename FromOp>
struct AllocLikeOpLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	protected:
	[[nodiscard]] virtual ArrayType getResultType(FromOp op) const = 0;
	[[nodiscard]] virtual mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, FromOp op, mlir::Value sizeBytes) const = 0;

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

	[[nodiscard]] ArrayType getResultType(AllocaOp op) const override
	{
		return op.resultType();
	}

	[[nodiscard]] mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocaOp op, mlir::Value sizeBytes) const override
	{
		auto typeConverter = this->getTypeConverter();
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.resultType().getElementType()));
		return rewriter.create<mlir::LLVM::AllocaOp>(loc, bufferPtrType, sizeBytes, op->getAttrs());
	}
};

class AllocOpLowering : public AllocLikeOpLowering<AllocOp>
{
	using AllocLikeOpLowering<AllocOp>::AllocLikeOpLowering;

	[[nodiscard]] ArrayType getResultType(AllocOp op) const override
	{
		return op.resultType();
	}

	[[nodiscard]] mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocOp op, mlir::Value sizeBytes) const override
	{
		// Insert the "malloc" declaration if it is not already present in the module
		auto heapAllocFunc = lookupOrCreateHeapAllocFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

		// Allocate the buffer
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.resultType().getElementType()));
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
		ArrayDescriptor descriptor(typeConverter, adaptor.memory());
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
		ArrayDescriptor descriptor(this->getTypeConverter(), adaptor.memory());
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

		auto sourceArrayType = op.source().getType().cast<ArrayType>();
		auto resultArrayType = op.resultType();

		ArrayDescriptor sourceDescriptor(typeConverter, adaptor.source());
		ArrayDescriptor result = ArrayDescriptor::undef(rewriter, typeConverter, loc, convertType(resultArrayType));

		mlir::Value index = adaptor.indexes()[0];

		for (size_t i = 1, e = sourceArrayType.getRank(); i < e; ++i)
		{
			mlir::Value size = sourceDescriptor.getSize(rewriter, loc, i);
			index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);

			if (i < adaptor.indexes().size())
			{
				mlir::Value offset = adaptor.indexes()[i];
				index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, offset);
			}
		}

		mlir::Value base = sourceDescriptor.getPtr(rewriter, loc);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);
		result.setPtr(rewriter, loc, ptr);

		mlir::Type rankType = result.getRankType();
		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, rankType, rewriter.getIntegerAttr(rankType, op.resultType().getRank()));
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
		auto indexes = adaptor.indexes();

		ArrayType arrayType = op.getArrayType();
		assert(arrayType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		ArrayDescriptor descriptor(typeConverter, adaptor.memory());
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
		auto indexes = adaptor.indexes();

		ArrayType arrayType = op.getArrayType();
		assert(arrayType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		ArrayDescriptor memoryDescriptor(this->getTypeConverter(), adaptor.memory());

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
		mlir::Type destination = op.resultType();

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
		mlir::Type destination = op.resultType();

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
		mlir::Type destination = op.resultType();

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
		mlir::Type destination = op.resultType();

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

struct ArrayCastOpLowering : public ModelicaOpConversion<ArrayCastOp>
{
	using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ArrayCastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor transformed(operands);
		mlir::Type source = op.memory().getType();
		mlir::Type destination = op.resultType();

		if (source.isa<ArrayType>())
		{
			if (auto resultType = destination.dyn_cast<ArrayType>())
			{
				rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, resultType, op.memory());
				return mlir::success();
			}

			if (auto resultType = destination.dyn_cast<UnsizedArrayType>())
			{
				ArrayDescriptor sourceDescriptor(this->getTypeConverter(), transformed.memory());

				// Create the unsized array descriptor that holds the ranked one.
				// The inner descriptor is allocated on stack.
				UnsizedArrayDescriptor resultDescriptor = UnsizedArrayDescriptor::undef(rewriter, loc, convertType(resultType));
				resultDescriptor.setRank(rewriter, loc, sourceDescriptor.getRank(rewriter, loc));

				mlir::Value underlyingDescPtr = rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), sourceDescriptor.computeSize(rewriter, loc), llvm::None);
				resultDescriptor.setPtr(rewriter, loc, underlyingDescPtr);
				mlir::Type sourceDescriptorArrayType = mlir::LLVM::LLVMPointerType::get(transformed.memory().getType());
				underlyingDescPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, sourceDescriptorArrayType, underlyingDescPtr);

				mlir::Type indexType = convertType(rewriter.getIndexType());
				mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));
				mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, underlyingDescPtr.getType(), underlyingDescPtr, zero);
				rewriter.create<mlir::LLVM::StoreOp>(loc, *sourceDescriptor, ptr);

				mlir::Value result = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, *resultDescriptor);
				rewriter.replaceOp(op, result);
				return mlir::success();
			}

			if (auto resultType = destination.dyn_cast<OpaquePointerType>())
			{
				ArrayDescriptor descriptor(this->getTypeConverter(), transformed.memory());
				mlir::Value result = rewriter.create<mlir::LLVM::BitcastOp>(loc, convertType(resultType), descriptor.getPtr(rewriter, loc));
				result = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, result);
				rewriter.replaceOp(op, result);
				return mlir::success();
			}
		}

		if (source.isa<OpaquePointerType>())
		{
			if (auto resultType = destination.dyn_cast<ArrayType>())
			{
				mlir::Type indexType = convertType(rewriter.getIndexType());
				auto typeConverter = this->getTypeConverter();

				ArrayDescriptor descriptor =
						ArrayDescriptor::undef(rewriter, typeConverter, loc, convertType(resultType));

				mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(convertType(resultType.getElementType()));
				mlir::Value ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, ptrType, op.memory());
				descriptor.setPtr(rewriter, loc, ptr);

				mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIntegerAttr(descriptor.getRankType(), resultType.getRank()));
				descriptor.setRank(rewriter, loc, rank);

				auto shape = resultType.getShape();
				llvm::SmallVector<mlir::Value, 3> sizes;

				for (auto size : shape)
				{
					assert(size != -1);
					sizes.push_back(rewriter.create<mlir::LLVM::ConstantOp>(
							loc, indexType, rewriter.getI64IntegerAttr(resultType.getRank())));
				}

				for (auto size : llvm::enumerate(sizes))
					descriptor.setSize(rewriter, loc, size.index(), size.value());

				rewriter.replaceOp(op, *descriptor);
				return mlir::success();
			}

			if (destination.isa<OpaquePointerType>())
			{
				rewriter.replaceOp(op, op.memory());
				return mlir::success();
			}
		}

		return rewriter.notifyMatchFailure(op, "Unknown conversion");
	}
};

struct UnrealizedCastOpLowering : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>
{
	using mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, mlir::PatternRewriter& rewriter) const override {
		rewriter.replaceOp(op, op->getOperands());
		return mlir::success();
	}
};

struct FuncAddressOfOpLowering : public mlir::ConvertOpToLLVMPattern<ida::FuncAddressOfOp>
{
	using mlir::ConvertOpToLLVMPattern<ida::FuncAddressOfOp>::ConvertOpToLLVMPattern;

	mlir::LogicalResult matchAndRewrite(ida::FuncAddressOfOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();

		if (mlir::LLVM::LLVMFuncOp function = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(op.callee()))
		{
			assert(function.getNumResults() == 1);
			rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, function);
			return mlir::success();
		}

		return mlir::failure();
	}
};

struct LoadPointerOpLowering : public mlir::ConvertOpToLLVMPattern<ida::LoadPointerOp>
{
	using mlir::ConvertOpToLLVMPattern<ida::LoadPointerOp>::ConvertOpToLLVMPattern;

	mlir::LogicalResult matchAndRewrite(ida::LoadPointerOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Type pointerType = getTypeConverter()->convertType(op.pointer().getType());

		mlir::Value indArray = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), pointerType, op.pointer(), op.offset());
		rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, indArray);

		return mlir::success();
	}
};

static void populateModelicaToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::OwningRewritePatternList& patterns)
{
	patterns.insert<
	    PackOpLowering,
			ExtractOpLowering,
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
			ArrayCastOpLowering,
			FuncAddressOfOpLowering,
			LoadPointerOpLowering>(typeConverter);
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

		if (failed(stdToLLVMConversionPass(module)))
		{
			mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
			signalPassFailure();
			return;
		}

		if (failed(castsFolderPass(module)))
		{
			mlir::emitError(module.getLoc(), "Error in folding the casts operations\n");
			signalPassFailure();
			return;
		}

		if (options.emitCWrappers)
		{
			if (failed(emitCWrappers(module)))
			{
				mlir::emitError(module.getLoc(), "Error in emitting the C wrappers\n");
				signalPassFailure();
				return;
			}
		}
	}

	private:
	mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module)
	{
		mlir::LowerToLLVMOptions llvmOptions(&getContext());
		llvmOptions.emitCWrappers = options.emitCWrappers;
		marco::codegen::TypeConverter typeConverter(&getContext(), llvmOptions, bitWidth);

		mlir::ConversionTarget target(getContext());
		target.addIllegalDialect<ModelicaDialect, ida::IdaDialect, mlir::StandardOpsDialect>();
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

	mlir::LogicalResult castsFolderPass(mlir::ModuleOp module)
	{
		mlir::ConversionTarget target(getContext());

		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<UnrealizedCastOpLowering>(&getContext());

		return applyFullConversion(module, target, std::move(patterns));
	}

	mlir::LogicalResult emitCWrappers(mlir::ModuleOp module)
	{
		bool success = true;
		mlir::OpBuilder builder(module);

		module.walk([&](mlir::LLVM::LLVMFuncOp function) {
			if (function.isExternal())
				return;

			builder.setInsertionPointAfter(function);

			llvm::SmallVector<mlir::Type, 3> wrapperArgumentsTypes;

			// Keep track for each original argument of its destination position
			llvm::SmallVector<long, 3> argumentsMapping;

			// Keep track for each original result if it has been moved to the
			// arguments list or not.
			bool resultMoved = false;

			mlir::Type wrapperResultType = function.getType().getReturnType();

			if (wrapperResultType.isa<mlir::LLVM::LLVMStructType>())
			{
				wrapperArgumentsTypes.push_back(mlir::LLVM::LLVMPointerType::get(wrapperResultType));
				wrapperResultType = mlir::LLVM::LLVMVoidType::get(function->getContext());
				resultMoved = true;
			}

			for (mlir::Type type : function.getType().getParams())
			{
				argumentsMapping.push_back(wrapperArgumentsTypes.size());

				if (type.isa<mlir::LLVM::LLVMStructType>())
					wrapperArgumentsTypes.push_back(mlir::LLVM::LLVMPointerType::get(type));
				else
					wrapperArgumentsTypes.push_back(type);
			}

			auto functionType = mlir::LLVM::LLVMFunctionType::get(wrapperResultType, wrapperArgumentsTypes, function.getType().isVarArg());
			auto wrapper = builder.create<mlir::LLVM::LLVMFuncOp>(function.getLoc(), ("__modelica_ciface_" + function.getName()).str(), functionType);
			mlir::Block* body = wrapper.addEntryBlock();
			builder.setInsertionPointToStart(body);

			llvm::SmallVector<mlir::Value, 3> args;
			llvm::SmallVector<mlir::Value, 1> results;

			for (auto type : llvm::enumerate(function.getArgumentTypes()))
			{
				mlir::Value wrapperArg = wrapper.getArgument(argumentsMapping[type.index()]);

				if (type.value().isa<mlir::LLVM::LLVMStructType>())
					args.push_back(builder.create<mlir::LLVM::LoadOp>(wrapper->getLoc(), wrapperArg));
				else
					args.push_back(wrapperArg);
			}

			auto call = builder.create<mlir::LLVM::CallOp>(wrapper->getLoc(), function, args);
			assert(call.getNumResults() <= 1);

			if (call->getNumResults() == 1)
			{
				mlir::Value result = call.getResult(0);

				if (resultMoved)
					builder.create<mlir::LLVM::StoreOp>(wrapper.getLoc(), result, wrapper.getArgument(0));
				else
					results.push_back(result);
			}

			builder.create<mlir::LLVM::ReturnOp>(wrapper->getLoc(), results);
		});

		return mlir::success(success);
	}

	ModelicaToLLVMConversionOptions options;
	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> marco::codegen::createLLVMLoweringPass(ModelicaToLLVMConversionOptions options, unsigned int bitWidth)
{
	return std::make_unique<LLVMLoweringPass>(options, bitWidth);
}
