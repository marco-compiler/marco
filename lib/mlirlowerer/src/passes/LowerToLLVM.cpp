#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <modelica/mlirlowerer/passes/LowerToLLVM.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen;

/**
 * Helper class to produce LLVM dialect operations extracting or inserting
 * values to a struct.
 */
class MemoryDescriptor {
	public:
	explicit MemoryDescriptor(mlir::Value value)
			: value(value),
				descriptorType(value.getType())
	{
		assert(value != nullptr && "Value cannot be null");
		assert(descriptorType.isa<mlir::LLVM::LLVMStructType>() && "Expected LLVM struct type");

		indexType = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
	}

	/**
	 * Allocate an empty descriptor.
	 *
	 * @param builder					operation builder
	 * @param location  			source location
	 * @param descriptorType	descriptor type
	 * @return descriptor
	 */
	static MemoryDescriptor undef(mlir::OpBuilder& builder, mlir::Location location, mlir::Type descriptorType)
	{
		mlir::Value descriptor = builder.create<mlir::LLVM::UndefOp>(location, descriptorType);
		return MemoryDescriptor(descriptor);
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
	 * Build IR to set the rank
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
		mlir::Type sizesContainerType = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[2];
		mlir::Value sizes = builder.create<mlir::LLVM::ExtractValueOp>(location, sizesContainerType, value, builder.getIndexArrayAttr(2));

		// Copy size values to stack-allocated memory
		mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(location, indexType, builder.getIntegerAttr(indexType, 1));
		mlir::Value sizesPtr = builder.create<mlir::LLVM::AllocaOp>(location, mlir::LLVM::LLVMPointerType::get(sizesContainerType), one, 0);
		builder.create<mlir::LLVM::StoreOp>(location, sizes, sizesPtr);

		// Load an return size value of interest
		mlir::Type sizeType = sizesContainerType.cast<mlir::LLVM::LLVMArrayType>().getElementType();
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

	private:
	mlir::Value value;
	mlir::Type descriptorType;
	mlir::Type indexType;
};

/**
 * Generic conversion pattern that provides some utility functions.
 *
 * @tparam FromOp type of the operation to be converted
 */
template<typename FromOp>
class ModelicaOpConversion : public mlir::OpConversionPattern<FromOp> {
	protected:
	using Adaptor = typename FromOp::Adaptor;
	using mlir::OpConversionPattern<FromOp>::OpConversionPattern;

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

	[[nodiscard]] mlir::Type voidPtrType(mlir::MLIRContext* context) const
	{
		return mlir::LLVM::LLVMPointerType::get(
				this->getTypeConverter()->convertType(IntegerType::get(context, 8)));
	}
};

template<typename FromOp>
struct AllocLikeOpLowering : public ModelicaOpConversion<FromOp>
{
	using ModelicaOpConversion<FromOp>::ModelicaOpConversion;

	protected:
	[[nodiscard]] virtual PointerType getResultType(FromOp op) const = 0;
	[[nodiscard]] virtual mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, FromOp op, mlir::Value sizeBytes) const = 0;

	private:
	mlir::LogicalResult matchAndRewrite(FromOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		auto pointerType = getResultType(op);

		mlir::Type indexType = this->convertType(rewriter.getIndexType());
		auto shape = pointerType.getShape();
		llvm::SmallVector<mlir::Value, 3> sizes;

		// Multi-dimensional arrays must be flattened into a 1-dimensional one-
		// For example, v[s1][s2][s3] becomes v[s1 * s2 * s3] and the access rule is that
		// v[i][j][k] = v[(i * s1 + j) * s2 + k].

		mlir::Value totalSize = createIndexConstant(rewriter, loc, 1);

		for (size_t i = 0, dynamicDimensions = 0, end = shape.size(); i < end; ++i)
		{
			long dimension = shape[i];

			if (dimension == -1)
				sizes.push_back(operands[dynamicDimensions++]);
			else
			{
				mlir::Value size = createIndexConstant(rewriter, loc, dimension);
				sizes.push_back(size);
			}

			totalSize = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, totalSize, sizes[i]);
		}

		// Buffer size in bytes
		mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(this->convertType(pointerType.getElementType()));
		mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
		mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(
				loc, elementPtrType, llvm::ArrayRef<mlir::Value>{nullPtr, totalSize});
		mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, gepPtr);

		// Allocate the underlying buffer
		mlir::Value buffer = allocateBuffer(rewriter, loc, op, sizeBytes);

		// Create the descriptor
		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getI64IntegerAttr(pointerType.getRank()));
		auto descriptorType = this->convertType(pointerType);
		auto descriptor = MemoryDescriptor::undef(rewriter, loc, descriptorType);

		descriptor.setPtr(rewriter, loc, buffer);
		descriptor.setRank(rewriter, loc, rank);

		for (size_t i = 0; i < sizes.size(); ++i)
			descriptor.setSize(rewriter, loc, i, sizes[i]);

		rewriter.replaceOp(op, *descriptor);
		return mlir::success();
	}

	mlir::Value createIndexConstant(mlir::OpBuilder& builder, mlir::Location loc, unsigned int value) const
	{
		mlir::Type indexType = this->convertType(builder.getIndexType());
		return builder.create<mlir::LLVM::ConstantOp>(loc, indexType, builder.getIndexAttr(value));
	}
};

class AllocaOpLowering : public AllocLikeOpLowering<AllocaOp>
{
	using AllocLikeOpLowering<AllocaOp>::AllocLikeOpLowering;

	[[nodiscard]] PointerType getResultType(AllocaOp op) const override
	{
		return op.resultType();
	}

	[[nodiscard]] mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocaOp op, mlir::Value sizeBytes) const override
	{
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.resultType().getElementType()));
		return rewriter.create<mlir::LLVM::AllocaOp>(loc, bufferPtrType, sizeBytes, op->getAttrs());
	}
};

class AllocOpLowering : public AllocLikeOpLowering<AllocOp>
{
	using AllocLikeOpLowering<AllocOp>::AllocLikeOpLowering;

	[[nodiscard]] PointerType getResultType(AllocOp op) const override
	{
		return op.resultType();
	}

	[[nodiscard]] mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocOp op, mlir::Value sizeBytes) const override
	{
		mlir::Type indexType = convertType(rewriter.getIndexType());

		// Insert the "malloc" declaration if it is not already present in the module
		auto mallocFunc = mlir::LLVM::lookupOrCreateMallocFn(op->getParentOfType<mlir::ModuleOp>(), indexType);

		// Allocate the buffer
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.resultType().getElementType()));
		auto results = createLLVMCall(rewriter, loc, mallocFunc, sizeBytes, voidPtrType(rewriter.getContext()));
		return rewriter.create<mlir::LLVM::BitcastOp>(loc, bufferPtrType, results[0]);
	}
};

class FreeOpLowering: public ModelicaOpConversion<FreeOp>
{
	using ModelicaOpConversion<FreeOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FreeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		// Insert the "free" declaration if it is not already present in the module
		auto freeFunc = mlir::LLVM::lookupOrCreateFreeFn(op->getParentOfType<mlir::ModuleOp>());

		// Extract the buffer address and call the "free" function
		MemoryDescriptor descriptor(adaptor.memory());
		mlir::Value address = descriptor.getPtr(rewriter, location);
		mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(location, voidPtrType(rewriter.getContext()), address);
		rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, llvm::None, rewriter.getSymbolRefAttr(freeFunc), casted);

		return mlir::success();
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
		MemoryDescriptor descriptor(adaptor.memory());
		mlir::Value size = descriptor.getSize(rewriter, location, adaptor.dimension());

		rewriter.replaceOp(op, size);
		return mlir::success();
	}
};

class SubscriptOpLowering: public ModelicaOpConversion<SubscriptionOp>
{
	using ModelicaOpConversion<SubscriptionOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SubscriptionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Type indexType = convertType(rewriter.getIndexType());

		auto sourcePointerType = op.source().getType().cast<PointerType>();
		auto resultPointerType = op.resultType();

		MemoryDescriptor sourceDescriptor(adaptor.source());
		MemoryDescriptor result = MemoryDescriptor::undef(rewriter, location, convertType(resultPointerType));

		mlir::Value index = adaptor.indexes()[0];

		for (size_t i = 1, e = sourcePointerType.getRank(); i < e; ++i)
		{
			mlir::Value size = sourceDescriptor.getSize(rewriter, location, i);
			index = rewriter.create<mlir::LLVM::MulOp>(location, indexType, index, size);

			if (i < adaptor.indexes().size())
				index = rewriter.create<mlir::LLVM::AddOp>(location, indexType, index, adaptor.indexes()[i]);
		}

		mlir::Value base = sourceDescriptor.getPtr(rewriter, location);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(location, base.getType(), base, index);
		result.setPtr(rewriter, location, ptr);

		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getI64IntegerAttr(op.resultType().getRank()));
		result.setRank(rewriter, location, rank);

		for (size_t i = sourcePointerType.getRank() - resultPointerType.getRank(), e = sourcePointerType.getRank(), j = 0; i < e; ++i, ++j)
			result.setSize(rewriter, location, j, sourceDescriptor.getSize(rewriter, location, i));

		rewriter.replaceOp(op, *result);
		return mlir::success();
	}
};

class LoadOpLowering: public ModelicaOpConversion<LoadOp>
{
	using ModelicaOpConversion<LoadOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LoadOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		Adaptor adaptor(operands);
		auto indexes = adaptor.indexes();

		PointerType pointerType = op.getPointerType();
		assert(pointerType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		MemoryDescriptor memoryDescriptor(adaptor.memory());

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

		PointerType pointerType = op.getPointerType();
		assert(pointerType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		MemoryDescriptor memoryDescriptor(adaptor.memory());

		auto indexType = convertType(rewriter.getIndexType());
		mlir::Value index = indexes.empty() ? rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0)) : indexes[0];

		for (size_t i = 1, e = indexes.size(); i < e; ++i)
		{
			mlir::Value size = memoryDescriptor.getSize(rewriter, loc, i);
			index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);
			index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, indexes[i]);

			//rewriter.create<PrintOp>(location, index); // TODO remove
		}

		mlir::Value base = memoryDescriptor.getPtr(rewriter, loc);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

		// Store the value
		rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), ptr);

		return mlir::success();
	}
};

class PrintOpLowering: public ModelicaOpConversion<PrintOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(PrintOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location loc = op.getLoc();
		auto module = op->getParentOfType<mlir::ModuleOp>();

		auto printfRef = getOrInsertPrintf(rewriter, module);
		mlir::Value semicolonCst = getOrCreateGlobalString(loc, rewriter, "semicolon", mlir::StringRef(";\0", 2), module);
		mlir::Value newLineCst = getOrCreateGlobalString(loc, rewriter, "newline", mlir::StringRef("\n\0", 2), module);

		for (auto value : Adaptor(operands).values())
			printElement(rewriter, value, semicolonCst, module);

		rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), printfRef, newLineCst);

		rewriter.eraseOp(op);
		return mlir::success();
	}

	mlir::Value getOrCreateGlobalString(mlir::Location loc, mlir::OpBuilder &builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) const
	{
		// Create the global at the entry of the module
		mlir::LLVM::GlobalOp global;

		if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name)))
		{
			mlir::OpBuilder::InsertionGuard insertGuard(builder);
			builder.setInsertionPointToStart(module.getBody());
			auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
			global = builder.create<mlir::LLVM::GlobalOp>(loc, type, true, mlir::LLVM::Linkage::Internal, name, builder.getStringAttr(value));
		}

		// Get the pointer to the first character in the global string
		mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);

		mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
				loc,
				mlir::IntegerType::get(builder.getContext(), 64),
				builder.getIntegerAttr(builder.getIndexType(), 0));

		return builder.create<mlir::LLVM::GEPOp>(
				loc,
				mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)),
				globalPtr, llvm::ArrayRef<mlir::Value>({cst0, cst0}));
	}

	mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& rewriter, mlir::ModuleOp module) const
	{
		auto *context = module.getContext();

		if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
			return foo;

		// Create a function declaration for printf, the signature is:
		//   * `i32 (i8*, ...)`
		auto llvmI32Ty = mlir::IntegerType::get(context, 32);
		auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
		auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

		// Insert the printf function into the body of the parent module.
		mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
		rewriter.setInsertionPointToStart(module.getBody());
		return rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
	}

	void printElement(mlir::OpBuilder& builder, mlir::Value value, mlir::Value separator, mlir::ModuleOp module) const
	{
		auto printfRef = getOrInsertPrintf(builder, module);

		mlir::Type type = value.getType();

		// Check if the separator should be printed
		builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, separator);

		mlir::Value formatSpecifier;

		if (type.isa<mlir::IntegerType>())
			formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_int", mlir::StringRef("%ld\0", 4), module);
		else if (type.isa<mlir::FloatType>())
			formatSpecifier = getOrCreateGlobalString(value.getLoc(), builder, "frmt_spec_float", mlir::StringRef("%.12f\0", 6), module);
		else
			assert(false && "Unknown type");

		builder.create<mlir::LLVM::CallOp>(value.getLoc(), printfRef, mlir::ValueRange({ formatSpecifier, value }));
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
			rewriter.replaceOpWithNewOp<mlir::IndexCastOp>(op, op.value(), getTypeConverter()->convertType(destination));
			return mlir::success();
		}

		if (destination.isa<RealType>())
		{
			unsigned int destinationBitWidth = destination.cast<RealType>().getBitWidth();
			mlir::Value value = rewriter.create<mlir::IndexCastOp>(location, op.value(), convertType(IntegerType::get(rewriter.getContext(), destinationBitWidth)));
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
			unsigned int destinationBitWidth = destination.cast<RealType>().getBitWidth();

			if (destinationBitWidth > 1)
				value = rewriter.create<mlir::LLVM::SExtOp>(location, convertType(IntegerType::get(rewriter.getContext(), destinationBitWidth)), value);

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

		mlir::Location location = op.getLoc();
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
			unsigned int destinationBitWidth = destination.cast<RealType>().getBitWidth();

			if (source.getBitWidth() < destinationBitWidth)
				value = rewriter.create<mlir::LLVM::SExtOp>(location, convertType(IntegerType::get(rewriter.getContext(), destinationBitWidth)), value);
			else if (source.getBitWidth() > destinationBitWidth)
				value = rewriter.create<mlir::LLVM::TruncOp>(location, convertType(IntegerType::get(rewriter.getContext(), destinationBitWidth)), value);

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
			unsigned int destinationBitWidth = destination.cast<IntegerType>().getBitWidth();

			if (source.getBitWidth() < destinationBitWidth)
				value = rewriter.create<mlir::LLVM::FPExtOp>(location, convertType(RealType::get(rewriter.getContext(), destinationBitWidth)), value);
			else if (source.getBitWidth() > destinationBitWidth)
				value = rewriter.create<mlir::LLVM::FPTruncOp>(location, convertType(RealType::get(rewriter.getContext(), destinationBitWidth)), value);

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

class CastCommonOpLowering: public ModelicaOpConversion<CastCommonOp>
{
	using ModelicaOpConversion<CastCommonOp>::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastCommonOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		llvm::SmallVector<mlir::Value, 3> values;

		for (auto tuple : llvm::zip(op->getOperands(), op->getResultTypes()))
		{
			mlir::Value castedValue = rewriter.create<CastOp>(location, std::get<0>(tuple), std::get<1>(tuple));
			values.push_back(castedValue);
		}

		rewriter.replaceOp(op, values);
		return mlir::success();
	}
};

struct PtrCastOpLowering : public mlir::OpRewritePattern<PtrCastOp>
{
	using mlir::OpRewritePattern<PtrCastOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(PtrCastOp op, mlir::PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, op.resultType(), op.memory());
		return mlir::success();
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

static void populateModelicaToLLVMConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, mlir::TypeConverter& typeConverter)
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
			CastCommonOpLowering,
			PrintOpLowering>(typeConverter, context); // TODO remove

	patterns.insert<PtrCastOpLowering>(context);
}

class LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit LLVMLoweringPass(ModelicaToLLVMConversionOptions options)
			: options(std::move(options))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
		registry.insert<mlir::LLVM::LLVMDialect>();
	}

	mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module)
	{
		mlir::ConversionTarget target(getContext());
		target.addIllegalDialect<ModelicaDialect, mlir::StandardOpsDialect>();
		target.addIllegalOp<mlir::FuncOp>();

		target.addLegalDialect<mlir::LLVM::LLVMDialect>();
		target.addIllegalOp<mlir::LLVM::DialectCastOp>();
		target.addLegalOp<mlir::UnrealizedConversionCastOp>();
		target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

		mlir::LowerToLLVMOptions llvmOptions;
		llvmOptions.emitCWrappers = options.emitCWrappers;

		modelica::codegen::TypeConverter typeConverter(&getContext(), llvmOptions);

		target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp>([&](mlir::Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
		target.addLegalOp<mlir::omp::TerminatorOp, mlir::omp::TaskyieldOp, mlir::omp::FlushOp, mlir::omp::BarrierOp, mlir::omp::TaskwaitOp>();

		mlir::OwningRewritePatternList patterns;
		populateModelicaToLLVMConversionPatterns(patterns, &getContext(), typeConverter);
		mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
		mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

		return applyPartialConversion(module, target, std::move(patterns));
	}

	mlir::LogicalResult castsFolderPass(mlir::ModuleOp module)
	{
		mlir::ConversionTarget target(getContext());
		target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
		target.addLegalDialect<mlir::omp::OpenMPDialect>();
		target.addLegalDialect<mlir::LLVM::LLVMDialect>();
		target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

		mlir::OwningRewritePatternList patterns;
		patterns.insert<UnrealizedCastOpLowering>(&getContext());

		return applyFullConversion(module, target, std::move(patterns));
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
	}

	private:
	ModelicaToLLVMConversionOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createLLVMLoweringPass(ModelicaToLLVMConversionOptions options)
{
	return std::make_unique<LLVMLoweringPass>(options);
}
