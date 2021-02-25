#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/mlirlowerer/passes/LowerToLLVM.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica;

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
		mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(location, indexType, builder.getIndexAttr(1));
		mlir::Value sizesPtr = builder.create<mlir::LLVM::AllocaOp>(location, mlir::LLVM::LLVMPointerType::get(sizesContainerType), one, 0);
		builder.create<mlir::LLVM::StoreOp>(location, sizes, sizesPtr);

		// Load an return size value of interest
		mlir::Type sizeType = sizesContainerType.cast<mlir::LLVM::LLVMArrayType>().getElementType();
		mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(location, indexType, builder.getIndexAttr(0));
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

	public:
	ModelicaOpConversion(mlir::MLIRContext* ctx, TypeConverter& typeConverter)
			: mlir::OpConversionPattern<FromOp>(typeConverter, ctx, 1)
	{
	}

	[[nodiscard]] modelica::TypeConverter& typeConverter() const {
		return *static_cast<modelica::TypeConverter *>(this->getTypeConverter());
	}

	[[nodiscard]] mlir::Type convertType(mlir::Type type) const {
		return typeConverter().convertType(type);
	}

	[[nodiscard]] bool isNumeric(mlir::Value value) const {
		return isNumericType(value.getType());
	}

	[[nodiscard]] bool isNumericType(mlir::Type type) const {
		return type.isa<IntegerType>() || type.isa<RealType>();
	}

	/**
	 * Get the type that can represent all the values with the minimum loss
	 * of information.
	 * Example: int, float -> float
	 *
	 * @param values  values
	 * @return most generic type
	 */
	[[nodiscard]] mlir::Type getMostGenericBaseType(mlir::ValueRange values) const
	{
		assert(!values.empty());
		mlir::Type resultType = nullptr;

		for (mlir::Value value : values)
		{
			mlir::Type type = value.getType();

			while (type.isa<PointerType>())
				type = type.cast<PointerType>().getElementType();

			if (resultType == nullptr)
			{
				resultType = type;
				continue;
			}

			if (resultType.isa<IntegerType>())
				resultType = type;
		}

		return resultType;
	}

	[[nodiscard]] llvm::SmallVector<mlir::Value, 3> getArrayDynamicDimensions(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array) const
	{
		assert(array.getType().isa<PointerType>());
		auto pointerType = array.getType().cast<PointerType>();
		auto shape = pointerType.getShape();

		llvm::SmallVector<mlir::Value, 3> dimensions;

		for (size_t i = 0, e = shape.size(); i < e; ++i)
		{
			if (shape[i] == -1)
			{
				mlir::Value dim = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(i));
				dimensions.push_back(builder.create<DimOp>(location, array, dim));
			}
		}

		return dimensions;
	}

	[[nodiscard]] mlir::Value getArrayTotalSize(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array, unsigned int rank) const
	{
		MemoryDescriptor descriptor(array);
		mlir::Value result = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(1));

		for (unsigned int dimension = 0; dimension < rank; ++dimension)
		{
			mlir::Value size = descriptor.getSize(builder, location, dimension);
			result = builder.create<mlir::MulIOp>(location, result, size);
		}

		return result;
	}

	/**
	 * Iterate over an array.
	 *
	 * @param builder   operation builder
	 * @param location  source location
	 * @param array     array to be iterated
	 * @param callback  function executed on each iteration
	 */
	void iterateArray(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array, std::function<void(mlir::ValueRange)> callback) const
	{
		assert(array.getType().isa<PointerType>());
		auto pointerType = array.getType().cast<PointerType>();

		mlir::Value zero = builder.create<mlir::ConstantOp>(location, builder.getZeroAttr(builder.getIndexType()));
		llvm::SmallVector<mlir::Value, 3> lowerBounds(pointerType.getRank(), zero);
		llvm::SmallVector<mlir::Value, 3> upperBounds;
		llvm::SmallVector<long, 3> steps;

		for (long dimension = 0; dimension < pointerType.getRank(); dimension++)
		{
			mlir::Value dim = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(dimension));
			upperBounds.push_back(builder.create<DimOp>(location, array, dim));
			steps.push_back(1);
		}

		// Create nested loops in order to iterate on each dimension of the array
		buildAffineLoopNest(
				builder, location, lowerBounds, upperBounds, steps,
				[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position) {
					callback(position);
				});
	}

	[[nodiscard]] mlir::Value allocateSameTypeArray(mlir::OpBuilder& builder, mlir::Location location, mlir::Value array) const
	{
		mlir::Type type = array.getType();
		assert(type.isa<PointerType>() && "Not an array");

		auto pointerType = type.cast<PointerType>();

		llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
		auto shape = pointerType.getShape();

		for (size_t i = 0, e = pointerType.getRank(); i < e; ++i)
		{
			if (shape[i] == -1)
			{
				mlir::Value dimension = builder.create<mlir::ConstantOp>(location, builder.getIndexAttr(i));
				dynamicDimensions.push_back(builder.create<DimOp>(location, array, dimension));
			}
		}

		return builder.create<AllocaOp>(location, pointerType.getElementType(), pointerType.getShape(), dynamicDimensions);
	}
};

class AllocaOpLowering: public ModelicaOpConversion<AllocaOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AllocaOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		mlir::Type indexType = typeConverter().indexType();
		auto shape = op.getPointerType().getShape();
		llvm::SmallVector<mlir::Value, 3> sizes;

		// Multi-dimensional arrays must be flattened into a 1-dimensional one-
		// For example, v[s1][s2][s3] becomes v[s1 * s2 * s3] and the access rule is that
		// v[i][j][k] = v[(i * s1 + j) * s2 + k].
		mlir::Value totalSize = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(1));

		for (size_t i = 0, dynamicDimensions = 0, end = shape.size(); i < end; ++i)
		{
			long dimension = shape[i];

			if (dimension == -1)
				sizes.push_back(operands[dynamicDimensions++]);
			else
			{
				mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(dimension));
				sizes.push_back(size);
			}

			totalSize = rewriter.create<mlir::LLVM::MulOp>(location, indexType, totalSize, sizes[i]);
		}

		// Allocate the buffer
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getPointerType().getElementType()));
		mlir::Value buffer = rewriter.create<mlir::LLVM::AllocaOp>(location, bufferPtrType, totalSize, op.getAttrs());

		// Create the descriptor
		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getI64IntegerAttr(op.getPointerType().getRank()));
		auto descriptorType = convertType(op.getPointerType());
		auto descriptor = MemoryDescriptor::undef(rewriter, location, descriptorType);

		descriptor.setPtr(rewriter, location, buffer);
		descriptor.setRank(rewriter, location, rank);

		for (size_t i = 0; i < sizes.size(); ++i)
			descriptor.setSize(rewriter, location, i, sizes[i]);

		rewriter.replaceOp(op, *descriptor);
		return mlir::success();
	}
};

class AllocOpLowering: public ModelicaOpConversion<AllocOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(AllocOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		auto indexType = typeConverter().indexType();
		auto shape = op.getPointerType().getShape();
		llvm::SmallVector<mlir::Value, 3> sizes;

		// Multi-dimensional arrays must be flattened into a 1-dimensional one-
		// For example, v[s1][s2][s3] becomes v[s1 * s2 * s3] and the access rule is that
		// v[i][j][k] = v[(i * s1 + j) * s2 + k].
		mlir::Value totalSize = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(1));

		for (size_t i = 0, dynamicDimensions = 0; i < shape.size(); ++i)
		{
			long dimension = shape[i];

			if (dimension == -1)
				sizes.push_back(operands[dynamicDimensions++]);
			else
			{
				mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(dimension));
				sizes.push_back(size);
			}

			totalSize = rewriter.create<mlir::LLVM::MulOp>(location, indexType, totalSize, sizes[i]);
		}

		// Insert the "malloc" declaration if it is not already present in the module
		auto mallocFunc = mlir::LLVM::lookupOrCreateMallocFn(op->getParentOfType<mlir::ModuleOp>(), indexType);

		// Allocate the buffer
		mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getPointerType().getElementType()));
		auto results = createLLVMCall(rewriter, location, mallocFunc, totalSize, typeConverter().voidPtrType());
		mlir::Value buffer = rewriter.create<mlir::LLVM::BitcastOp>(location, bufferPtrType, results[0]);

		// Create the descriptor
		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getI64IntegerAttr(op.getPointerType().getRank()));
		mlir::Type descriptorType = convertType(op.getPointerType());
		auto descriptor = MemoryDescriptor::undef(rewriter, location, descriptorType);

		descriptor.setPtr(rewriter, location, buffer);
		descriptor.setRank(rewriter, location, rank);

		for (size_t i = 0; i < sizes.size(); ++i)
			descriptor.setSize(rewriter, location, i, sizes[i]);

		rewriter.replaceOp(op, *descriptor);
		return mlir::success();
	}
};

class FreeOpLowering: public ModelicaOpConversion<FreeOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(FreeOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		// Insert the "free" declaration if it is not already present in the module
		auto freeFunc = mlir::LLVM::lookupOrCreateFreeFn(op->getParentOfType<mlir::ModuleOp>());

		// Extract the buffer address and call the "free" function
		MemoryDescriptor descriptor(adaptor.memory());
		mlir::Value address = descriptor.getPtr(rewriter, location);
		mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(location, typeConverter().voidPtrType(), address);
		rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, llvm::None, rewriter.getSymbolRefAttr(freeFunc), casted);

		return mlir::success();
	}
};

class DimOpLowering: public ModelicaOpConversion<DimOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

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
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(SubscriptionOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Type indexType = typeConverter().indexType();

		auto sourcePointerType = op.source().getType().cast<PointerType>();
		auto resultPointerType = op.getPointerType();

		MemoryDescriptor sourceDescriptor(adaptor.source());
		MemoryDescriptor result = MemoryDescriptor::undef(rewriter, location, convertType(resultPointerType));

		mlir::Value index = adaptor.indexes()[0];

		for (size_t i = 1, e = sourcePointerType.getRank(); i < e; ++i)
		{
			mlir::Value size = sourceDescriptor.getSize(rewriter, location, i);
			index = rewriter.create<mlir::LLVM::MulOp>(location, indexType, index, size);
		}

		mlir::Value base = sourceDescriptor.getPtr(rewriter, location);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(location, base.getType(), base, index);
		result.setPtr(rewriter, location, ptr);

		mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getI64IntegerAttr(op.getPointerType().getRank()));
		result.setRank(rewriter, location, rank);

		for (size_t i = sourcePointerType.getRank() - resultPointerType.getRank(), e = sourcePointerType.getRank(), j = 0; i < e; ++i, ++j)
			result.setSize(rewriter, location, j, sourceDescriptor.getSize(rewriter, location, i));

		rewriter.replaceOp(op, *result);
		return mlir::success();
	}
};

class LoadOpLowering: public ModelicaOpConversion<LoadOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LoadOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		Adaptor adaptor(operands);
		auto indexes = adaptor.indexes();

		PointerType pointerType = op.getPointerType();
		assert(pointerType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		MemoryDescriptor memoryDescriptor(adaptor.memory());

		auto indexType = typeConverter().indexType();
		mlir::Value index = indexes.empty() ? rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(0)) : indexes[0];

		for (size_t i = 1, e = indexes.size(); i < e; ++i)
		{
			mlir::Value size = memoryDescriptor.getSize(rewriter, location, i);
			index = rewriter.create<mlir::LLVM::MulOp>(location, indexType, index, size);
			index = rewriter.create<mlir::LLVM::AddOp>(location, indexType, index, indexes[i]);
		}

		mlir::Value base = memoryDescriptor.getPtr(rewriter, location);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(location, base.getType(), base, index);

		// Load the value
		rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptr);

		return mlir::success();
	}
};

class StoreOpLowering: public ModelicaOpConversion<StoreOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(StoreOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		Adaptor adaptor(operands);
		auto indexes = adaptor.indexes();

		PointerType pointerType = op.getPointerType();
		assert(pointerType.getRank() == indexes.size() && "Wrong indexes amount");

		// Determine the address into which the value has to be stored.
		MemoryDescriptor memoryDescriptor(adaptor.memory());

		auto indexType = typeConverter().indexType();
		mlir::Value index = indexes.empty() ? rewriter.create<mlir::LLVM::ConstantOp>(location, indexType, rewriter.getIndexAttr(0)) : indexes[0];

		for (size_t i = 1, e = indexes.size(); i < e; ++i)
		{
			mlir::Value size = memoryDescriptor.getSize(rewriter, location, i);
			index = rewriter.create<mlir::LLVM::MulOp>(location, indexType, index, size);
			index = rewriter.create<mlir::LLVM::AddOp>(location, indexType, index, indexes[i]);
		}

		mlir::Value base = memoryDescriptor.getPtr(rewriter, location);
		mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(location, base.getType(), base, index);

		// Store the value
		rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.value(), ptr);

		return mlir::success();
	}
};

class IfOpLowering: public ModelicaOpConversion<IfOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(IfOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		Adaptor adaptor(operands);
		bool hasElseBlock = !op.elseRegion().empty();

		mlir::scf::IfOp ifOp;

		// In order to move the blocks into the SCF operation, we need to override
		// its blocks builders. In fact, the default ones already place the
		// SCF::YieldOp terminators, but our IR already has the Modelica::YieldOps
		// converted to SCF::YieldOps (note that although in this context the
		// Modelica::YieldOps don't carry any useful data, we can't avoid creating
		// them, or the blocks would have no terminator, which is illegal).

		assert(op.thenRegion().getBlocks().size() == 1);

		auto thenBuilder = [&](mlir::OpBuilder& builder, mlir::Location location)
		{
			rewriter.mergeBlocks(&op.thenRegion().front(), rewriter.getInsertionBlock(), llvm::None);
		};

		if (hasElseBlock)
		{
			assert(op.elseRegion().getBlocks().size() == 1);

			ifOp = rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
					op, llvm::None, adaptor.condition(), thenBuilder,
					[&](mlir::OpBuilder& builder, mlir::Location location)
					{
						rewriter.mergeBlocks(&op.elseRegion().front(), builder.getInsertionBlock(), llvm::None);
					});
		}
		else
		{
			ifOp = rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(op, llvm::None, adaptor.condition(), thenBuilder, nullptr);
		}

		// Replace the Modelica::YieldOp terminator in the "then" branch with
		// a SCF::YieldOp.

		mlir::Block* thenBlock = &ifOp.thenRegion().front();
		auto thenTerminator = mlir::cast<YieldOp>(thenBlock->getTerminator());
		rewriter.setInsertionPointToEnd(thenBlock);
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(thenTerminator, thenTerminator.getOperands());

		// If the operation also has an "else" block, also replace its
		// Modelica::YieldOp terminator with a SCF::YieldOp.

		if (hasElseBlock)
		{
			mlir::Block* elseBlock = &ifOp.elseRegion().front();
			auto elseTerminator = mlir::cast<YieldOp>(elseBlock->getTerminator());
			rewriter.setInsertionPointToEnd(elseBlock);
			rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(elseTerminator, elseTerminator.getOperands());
		}

		return mlir::success();
	}
};

class ForOpLowering: public ModelicaOpConversion<ForOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(ForOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		// Split the current block
		mlir::Block* currentBlock = rewriter.getInsertionBlock();
		mlir::Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

		// Inline regions
		mlir::Block* conditionBlock = &op.condition().front();
		mlir::Block* bodyBlock = &op.body().front();
		mlir::Block* stepBlock = &op.step().front();

		rewriter.inlineRegionBefore(op.step(), continuation);
		rewriter.inlineRegionBefore(op.body(), stepBlock);
		rewriter.inlineRegionBefore(op.condition(), bodyBlock);

		// Start the for loop by branching to the "condition" region
		rewriter.setInsertionPointToEnd(currentBlock);
		rewriter.create<mlir::BranchOp>(location, conditionBlock, op.args());

		// The loop is supposed to be breakable. Thus, before checking the normal
		// condition, we first need to check if the break condition variable has
		// been set to true in the previous loop execution. If it is set to true,
		// it means that a break statement has been executed and thus the loop
		// must be terminated.

		rewriter.setInsertionPointToStart(conditionBlock);

		mlir::Value breakCondition = rewriter.create<LoadOp>(location, adaptor.breakCondition());
		mlir::Value returnCondition = rewriter.create<LoadOp>(location, adaptor.returnCondition());
		mlir::Value stopCondition = rewriter.create<mlir::OrOp>(location, breakCondition, returnCondition);

		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));
		mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

		auto ifOp = rewriter.create<mlir::scf::IfOp>(location, rewriter.getI1Type(), condition, true);
		mlir::Block* originalCondition = rewriter.splitBlock(conditionBlock, rewriter.getInsertionPoint());

		// If the break condition variable is set to true, return false from the
		// condition block in order to stop the loop execution.
		rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
		mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(false));
		rewriter.create<mlir::scf::YieldOp>(location, falseValue);

		// Move the original condition check in the "else" branch
		rewriter.mergeBlocks(originalCondition, &ifOp.elseRegion().front(), llvm::None);
		rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
		auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

		// The original condition operation is converted to the SCF one and takes
		// as condition argument the result of the If operation, which is false
		// if a break must be executed or the intended condition value otherwise.
		rewriter.setInsertionPointAfter(ifOp);
		rewriter.create<mlir::CondBranchOp>(location, ifOp.getResult(0), bodyBlock, conditionOp.args(), continuation, llvm::None);

		// Replace "body" block terminator with a branch to the "step" block
		rewriter.setInsertionPointToEnd(bodyBlock);
		auto bodyYieldOp = mlir::cast<YieldOp>(bodyBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp->getOperands());

		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepBlock);
		auto stepYieldOp = mlir::cast<YieldOp>(stepBlock->getTerminator());
		rewriter.replaceOpWithNewOp<mlir::BranchOp>(stepYieldOp, conditionBlock, stepYieldOp->getOperands());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

class WhileOpLowering: public ModelicaOpConversion<WhileOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(WhileOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		auto whileOp = rewriter.create<mlir::scf::WhileOp>(location, llvm::None, llvm::None);

		// The body block requires no modification apart from the change of the
		// terminator to the SCF dialect one.

		rewriter.createBlock(&whileOp.after());
		rewriter.mergeBlocks(&op.body().front(), &whileOp.after().front(), llvm::None);
		mlir::Block* body = &whileOp.after().front();
		auto bodyTerminator = mlir::cast<YieldOp>(body->getTerminator());
		rewriter.setInsertionPointToEnd(body);
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(bodyTerminator, bodyTerminator.getOperands());

		// The loop is supposed to be breakable. Thus, before checking the normal
		// condition, we first need to check if the break condition variable has
		// been set to true in the previous loop execution. If it is set to true,
		// it means that a break statement has been executed and thus the loop
		// must be terminated.

		rewriter.createBlock(&whileOp.before());
		rewriter.setInsertionPointToStart(&whileOp.before().front());

		mlir::Value breakCondition = rewriter.create<mlir::LoadOp>(location, adaptor.breakCondition());
		mlir::Value returnCondition = rewriter.create<mlir::LoadOp>(location, adaptor.returnCondition());
		mlir::Value stopCondition = rewriter.create<mlir::OrOp>(location, breakCondition, returnCondition);

		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));
		mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

		auto ifOp = rewriter.create<mlir::scf::IfOp>(location, rewriter.getI1Type(), condition, true);

		// If the break condition variable is set to true, return false from the
		// condition block in order to stop the loop execution.
		rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
		mlir::Value falseValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(false));
		rewriter.create<mlir::scf::YieldOp>(location, falseValue);

		// Move the original condition check in the "else" branch
		rewriter.mergeBlocks(&op.condition().front(), &ifOp.elseRegion().front(), llvm::None);
		rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
		auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
		rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

		// The original condition operation is converted to the SCF one and takes
		// as condition argument the result of the If operation, which is false
		// if a break must be executed or the intended condition value otherwise.
		rewriter.setInsertionPointAfter(ifOp);
		rewriter.create<mlir::scf::ConditionOp>(location, ifOp.getResult(0), conditionOp.args());

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

class CastOpLowering: public ModelicaOpConversion<CastOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(CastOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Type source = op.value().getType();
		mlir::Type destination = op.resultType();

		if (source == destination)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		mlir::Type sourceBase = source;
		mlir::Type destinationBase = destination;

		if (source.isa<PointerType>())
		{
			return rewriter.notifyMatchFailure(op, "Not supported yet");

			/*
			sourceBase = source.cast<PointerType>().getElementType();
			auto sourceShape = source.cast<PointerType>().getShape();

			if (destination.isa<PointerType>())
			{
				auto destinationShape = destination.cast<PointerType>().getShape();
				destinationBase = destination.cast<PointerType>().getElementType();
				assert(all_of(llvm::zip(sourceShape, destinationShape),
											[](const auto& pair)
											{
												return std::get<0>(pair) == std::get<1>(pair);
											}));

				destination = mlir::VectorType::get(destinationShape, destinationBase);
			}
			else
			{
				destination = mlir::VectorType::get(sourceShape, destinationBase);
			}
			 */
		}

		if (sourceBase == destinationBase)
		{
			rewriter.replaceOp(op, op.value());
			return mlir::success();
		}

		if (sourceBase.isa<IntegerType>())
		{
			if (destinationBase.isa<RealType>())
			{
				rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destinationBase), adaptor.value());
				return mlir::success();
			}
		}

		if (sourceBase.isa<RealType>())
		{
			if (destinationBase.isa<IntegerType>())
			{
				rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(op, convertType(destinationBase), adaptor.value());
				return mlir::success();
			}
		}

		return rewriter.notifyMatchFailure(op, "Unsupported type conversion");
	}
};

class CastCommonOpLowering: public ModelicaOpConversion<CastCommonOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

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

/**
 * Negate a boolean scalar.
 */
class NegateOpScalarLowering: public ModelicaOpConversion<NegateOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());
		auto type = adaptor.operand().getType();
		return mlir::success(type.isa<BooleanType>());
	}

	void rewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		// There is no native negate operation in LLVM IR, so we need to leverage
		// a property of the XOR operation: x XOR true = NOT x
		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));
		rewriter.replaceOpWithNewOp<mlir::XOrOp>(op, trueValue, adaptor.operand());
	}
};

/**
 * Negate a boolean array.
 */
class NegateOpArrayLowering: public ModelicaOpConversion<NegateOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());
		auto type = adaptor.operand().getType();

		if (!type.isa<PointerType>())
			return mlir::failure();

		auto pointerType = type.cast<PointerType>();
		return mlir::success(pointerType.getElementType().isa<BooleanType>());
	}

	void rewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value result = allocateSameTypeArray(rewriter, location, op.operand());
		rewriter.replaceOp(op, result);

		mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));

		iterateArray(rewriter, location, op.operand(),
								 [&](mlir::ValueRange position) {
									 mlir::Value value = rewriter.create<LoadOp>(location, op.operand(), position);
									 mlir::Value negated = rewriter.create<NegateOp>(location, value);
									 rewriter.create<StoreOp>(location, negated, result, position);
								 });
	}
};

class EqOpIntegerLowering: public ModelicaOpConversion<EqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(EqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::eq, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class EqOpRealLowering: public ModelicaOpConversion<EqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(EqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::oeq, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class EqOpMixedLowering: public ModelicaOpConversion<EqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(EqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::oeq, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class NotEqOpIntegerLowering: public ModelicaOpConversion<NotEqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(NotEqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::ne, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class NotEqOpRealLowering: public ModelicaOpConversion<NotEqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(NotEqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::one, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class NotEqOpMixedLowering: public ModelicaOpConversion<NotEqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(NotEqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::one, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GtOpIntegerLowering: public ModelicaOpConversion<GtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(GtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::sgt, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GtOpRealLowering: public ModelicaOpConversion<GtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(GtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::ogt, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GtOpMixedLowering: public ModelicaOpConversion<GtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(GtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::ogt, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GteOpIntegerLowering: public ModelicaOpConversion<GteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(GteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::sge, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GteOpRealLowering: public ModelicaOpConversion<GteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(GteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::oge, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class GteOpMixedLowering: public ModelicaOpConversion<GteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(GteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::oge, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LtOpIntegerLowering: public ModelicaOpConversion<LtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(LtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::slt, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LtOpRealLowering: public ModelicaOpConversion<LtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(LtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::olt, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LtOpMixedLowering: public ModelicaOpConversion<LtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(LtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::olt, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LteOpIntegerLowering: public ModelicaOpConversion<LteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<BooleanType>());

		if (lhsType.isa<IntegerType>())
			return mlir::success(rhsType.isa<IntegerType>());

		return mlir::failure();
	}

	void rewrite(LteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::ICmpOp>(location, mlir::LLVM::ICmpPredicate::sle, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LteOpRealLowering: public ModelicaOpConversion<LteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<RealType>())
			return mlir::success(rhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(LteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);
		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::ole, adaptor.lhs(), adaptor.rhs());
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

class LteOpMixedLowering: public ModelicaOpConversion<LteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		mlir::Type lhsType = adaptor.lhs().getType();
		mlir::Type rhsType = adaptor.rhs().getType();

		if (lhsType.isa<IntegerType>() || lhsType.isa<BooleanType>())
			return mlir::success(rhsType.isa<RealType>());

		if (rhsType.isa<IntegerType>() || rhsType.isa<BooleanType>())
			return mlir::success(lhsType.isa<RealType>());

		return mlir::failure();
	}

	void rewrite(LteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		if (lhs.getType().isa<RealType>())
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		else if (rhs.getType().isa<RealType>())
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());

		mlir::Value result = rewriter.create<mlir::LLVM::FCmpOp>(location, mlir::LLVM::FCmpPredicate::ole, lhs, rhs);
		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	}
};

/**
 * Sum of two numeric scalars.
 */
class AddOpScalarLowering: public ModelicaOpConversion<AddOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());
		return mlir::success(isNumericType(adaptor.lhs().getType()) && isNumericType(adaptor.rhs().getType()));
	}

	void rewrite(AddOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		auto castOp = rewriter.create<CastCommonOp>(location, op->getOperands());
		Adaptor adaptor(castOp.getResults());

		mlir::Type type = castOp.resultType();

		if (type.isa<IntegerType>())
			rewriter.replaceOpWithNewOp<mlir::AddIOp>(op, adaptor.lhs(), adaptor.rhs());
		else if (type.isa<RealType>())
			rewriter.replaceOpWithNewOp<mlir::AddFOp>(op, adaptor.lhs(), adaptor.rhs());
	}
};

/**
 * Sum of two numeric arrays.
 */
class AddOpArrayLowering: public ModelicaOpConversion<AddOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult match(mlir::Operation* op) const override
	{
		Adaptor adaptor(op->getOperands());

		if (!adaptor.lhs().getType().isa<PointerType>())
			return mlir::failure();

		if (!adaptor.rhs().getType().isa<PointerType>())
			return mlir::failure();

		auto lhsPointerType = adaptor.lhs().getType().cast<PointerType>();
		auto rhsPointerType = adaptor.rhs().getType().cast<PointerType>();

		for (auto pair : llvm::zip(lhsPointerType.getShape(), rhsPointerType.getShape()))
		{
			auto lhsDimension = std::get<0>(pair);
			auto rhsDimension = std::get<1>(pair);

			if (lhsDimension != -1 && rhsDimension != -1 && lhsDimension != rhsDimension)
				return mlir::failure();
		}

		if (!isNumericType(lhsPointerType.getElementType()))
			return mlir::failure();

		if (!isNumericType(rhsPointerType.getElementType()))
			return mlir::failure();

		return mlir::success();
	}

	void rewrite(AddOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();

		// Allocate the result array
		mlir::Type baseType = getMostGenericBaseType({ op.lhs(), op.rhs() });
		auto shape = op.lhs().getType().cast<PointerType>().getShape();
		auto dynamicDimensions = getArrayDynamicDimensions(rewriter, location, op.lhs());
		mlir::Value result = rewriter.create<AllocaOp>(location, baseType, shape, dynamicDimensions);

		// Sum each element
		iterateArray(rewriter, location, op.lhs(),
								 [&](mlir::ValueRange position) {
									 mlir::Value lhs = rewriter.create<LoadOp>(location, op.lhs(), position);
									 lhs = rewriter.create<CastOp>(location, lhs, baseType);

									 mlir::Value rhs = rewriter.create<LoadOp>(location, op.rhs(), position);
									 rhs = rewriter.create<CastOp>(location, rhs, baseType);

									 Adaptor adaptor({ lhs, rhs });
									 rewriter.create<AddOp>(location, lhs.getType(), adaptor.lhs(), adaptor.rhs());
								 });

		rewriter.replaceOp(op, result);
	}
};

void ModelicaToLLVMLoweringPass::getDependentDialects(mlir::DialectRegistry &registry) const {
	registry.insert<mlir::StandardOpsDialect>();
	registry.insert<mlir::LLVM::LLVMDialect>();
	registry.insert<mlir::AffineDialect>();
}

ModelicaToLLVMLoweringPass::ModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options)
		: options(std::move(options))
{
}

void ModelicaToLLVMLoweringPass::runOnOperation()
{
	auto module = getOperation();

	mlir::ConversionTarget target(getContext());
	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

	// We need to mark the scf::YieldOp and AffineYieldOp as legal due to a
	// current limitation of MLIR. In fact, they are used just as a placeholder
	// and would lead to conversion problems if encountered while lowering.
	target.addLegalOp<mlir::scf::YieldOp>();
	target.addLegalOp<mlir::AffineYieldOp>();

	// During this lowering, we will also be lowering the MemRef types, that are
	// currently being operated on, to a representation in LLVM. To perform this
	// conversion we use a TypeConverter as part of the lowering. This converter
	// details how one type maps to another. This is necessary now that we will be
	// doing more complicated lowerings, involving loop region arguments.
	mlir::LowerToLLVMOptions llvmLoweringOptions;
	llvmLoweringOptions.emitCWrappers = true;

	TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	populateStdToLLVMConversionPatterns(typeConverter, patterns);
	populateAffineToStdConversionPatterns(patterns, &getContext());
	populateLoopToStdConversionPatterns(patterns, &getContext());
	populateModelicaToLLVMConversionPatterns(patterns, &getContext(), typeConverter);

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyFullConversion(module, target, std::move(patterns))))
	{
		mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
		signalPassFailure();
	}
}

void modelica::populateModelicaToLLVMConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, modelica::TypeConverter& typeConverter)
{
	patterns.insert<AllocaOpLowering, AllocOpLowering, FreeOpLowering, SubscriptOpLowering, DimOpLowering, LoadOpLowering, StoreOpLowering>(context, typeConverter);
	patterns.insert<IfOpLowering, ForOpLowering, WhileOpLowering>(context, typeConverter);
	patterns.insert<CastOpLowering, CastCommonOpLowering>(context, typeConverter);

	// Logic operations
	patterns.insert<NegateOpScalarLowering, NegateOpArrayLowering>(context, typeConverter);
	patterns.insert<EqOpIntegerLowering, EqOpRealLowering, EqOpMixedLowering>(context, typeConverter);
	patterns.insert<NotEqOpIntegerLowering, NotEqOpRealLowering, NotEqOpMixedLowering>(context, typeConverter);
	patterns.insert<GtOpIntegerLowering, GtOpRealLowering, GtOpMixedLowering>(context, typeConverter);
	patterns.insert<GteOpIntegerLowering, GteOpRealLowering, GteOpMixedLowering>(context, typeConverter);
	patterns.insert<LtOpIntegerLowering, LtOpRealLowering, LtOpMixedLowering>(context, typeConverter);
	patterns.insert<LteOpIntegerLowering, LteOpRealLowering, LteOpMixedLowering>(context, typeConverter);

	// Math operations
	patterns.insert<AddOpScalarLowering, AddOpArrayLowering>(context, typeConverter);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options)
{
	return std::make_unique<ModelicaToLLVMLoweringPass>(options);
}
