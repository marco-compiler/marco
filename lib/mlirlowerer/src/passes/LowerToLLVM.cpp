#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
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
template <typename FromOp>
class ModelicaOpConversion : public mlir::OpConversionPattern<FromOp> {
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
		FreeOp::Adaptor adaptor(operands);

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
		DimOp::Adaptor adaptor(operands);

		// The actual size of each dimensions is stored in the memory description
		// structure.
		MemoryDescriptor descriptor(adaptor.memory());
		mlir::Value size = descriptor.getSize(rewriter, location, adaptor.dimension());

		rewriter.replaceOp(op, size);
		return mlir::success();
	}
};

class LoadOpLowering: public ModelicaOpConversion<LoadOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LoadOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		LoadOp::Adaptor adaptor(operands);
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
		StoreOp::Adaptor adaptor(operands);
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
		IfOp::Adaptor adaptor(operands);
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
		ForOp::Adaptor adaptor(operands);

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
		WhileOp::Adaptor adaptor(operands);

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

class NegateOpLowering: public ModelicaOpConversion<NegateOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NegateOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		NegateOp::Adaptor adaptor(operands);

		if (op.operand().getType().isa<PointerType>())
		{
			// Operand is an array
			auto pointerType = op.operand().getType().cast<PointerType>();

			if (pointerType.hasConstantShape())
			{
				// TODO: use SIMD instruction for static arrays
			}

			mlir::Value result = allocateSameTypeArray(rewriter, location, op.operand());

			//mlir::Value arraySize = getArrayTotalSize(rewriter, location, adaptor.operand(), pointerType.getRank());
			//mlir::Value zeroValue = rewriter.create<mlir::LLVM::ConstantOp>(location, typeConverter().indexType(), rewriter.getIndexAttr(0));
			// TODO: use llvm operations or higher ones?

			mlir::Value zero = rewriter.create<mlir::ConstantOp>(location, rewriter.getZeroAttr(rewriter.getIndexType()));
			llvm::SmallVector<mlir::Value, 3> lowerBounds(pointerType.getRank(), zero);
			llvm::SmallVector<mlir::Value, 3> upperBounds;
			llvm::SmallVector<long, 3> steps;

			for (long dimension = 0; dimension < pointerType.getRank(); dimension++)
			{
				mlir::Value dim = rewriter.create<mlir::ConstantOp>(location, rewriter.getIndexAttr(dimension));
				//upperBounds.push_back(rewriter.create<DimOp>(location, op.operand(), dim));
				upperBounds.push_back(rewriter.create<mlir::ConstantOp>(location, rewriter.getIndexAttr(3)));
				steps.push_back(1);
			}

			mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));

			/*
			buildAffineLoopNest(
					rewriter, location, lowerBounds, upperBounds, steps,
					[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
						nestedBuilder.create<StoreOp>(location, trueValue, result, ivs);
					});
			*/

			//mlir::Value memtmp = rewriter.create<mlir::AllocaOp>(location, mlir::MemRefType::get({}, rewriter.getI1Type()));
			buildAffineLoopNest(
					rewriter, location, lowerBounds, upperBounds, steps,
					[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
						//mlir::Value t = nestedBuilder.create<mlir::ConstantOp>(location, nestedBuilder.getBoolAttr(true));
						nestedBuilder.create<StoreOp>(location, trueValue, result, ivs);
					});

			rewriter.replaceOp(op, result);
		}
		else
		{
			// Operand is a scalar
			mlir::Value trueValue = rewriter.create<mlir::ConstantOp>(location, rewriter.getBoolAttr(true));
			rewriter.replaceOpWithNewOp<mlir::XOrOp>(op, trueValue, adaptor.operand());
		}

		op->getParentOp()->dump();

		return mlir::success();

		/*
		mlir::Location location = op.getLoc();
		mlir::Value operand = op->getOperand(0);

		mlir::Type type = op->getResultTypes()[0];

		if (type.isa<mlir::MemRefType>())
		{
			auto memRefType = type.cast<mlir::MemRefType>();

			mlir::Value fake = rewriter.create<AllocaOp>(location, MemRefType::get({ 2 }, rewriter.getI1Type()));
			rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true)).getResult(), fake, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0)).getResult());
			rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false)).getResult(), fake, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(1)).getResult());
			operand = fake;

			mlir::VectorType vectorType = VectorType::get(memRefType.getShape(), memRefType.getElementType());
			mlir::Value zeroValue = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0));
			SmallVector<mlir::Value, 3> indexes(memRefType.getRank(), zeroValue);
			mlir::Value vector = rewriter.create<AffineVectorLoadOp>(location, vectorType, operand, indexes);
			rewriter.create<mlir::vector::PrintOp>(location, vector);

			SmallVector<bool, 3> trueValues(memRefType.getNumElements(), true);
			mlir::Value trueVector = rewriter.create<ConstantOp>(location, rewriter.getBoolVectorAttr(trueValues));
			mlir::Value xorOp = rewriter.create<XOrOp>(location, vector, trueVector);
			rewriter.create<mlir::vector::PrintOp>(location, xorOp);

			mlir::Value destination = rewriter.create<AllocaOp>(location, memRefType);
			rewriter.create<AffineVectorStoreOp>(location, xorOp, destination, indexes);

			//mlir::Value unranked = rewriter.create<MemRefCastOp>(location, destination, MemRefType::get(-1, rewriter.getI32Type()));
			//rewriter.create<CallOp>(location, "print_memref_i32", TypeRange(), unranked);

			//rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false)).getResult(), destination, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0)).getResult());
			//rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true)).getResult(), destination, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(1)).getResult());
			//mlir::Value vectorAfterStore = rewriter.create<AffineVectorLoadOp>(location, vectorType, destination, indexes);
			//rewriter.create<mlir::vector::PrintOp>(location, vectorAfterStore);

			rewriter.replaceOp(op, destination);
		}
		else
		{
			mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
			rewriter.replaceOpWithNewOp<XOrOp>(op, operand, trueValue);
		}

		return mlir::success();
		*/
	}
};

class EqOpLowering: public ModelicaOpConversion<EqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(EqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		EqOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::eq, lhs, rhs);
		else if (lhsType.isa<RealType>() && rhsType.isa<RealType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OEQ, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhsType);
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OEQ, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhsType);
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OEQ, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

class NotEqOpLowering: public ModelicaOpConversion<NotEqOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(NotEqOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		NotEqOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::ne, lhs, rhs);
		else if (lhsType.isa<RealType>() && rhsType.isa<RealType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::ONE, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::ONE, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::ONE, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

class GtOpLowering: public ModelicaOpConversion<GtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(GtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		GtOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::sgt, lhs, rhs);
		else if (lhsType.isa<mlir::FloatType>() && rhsType.isa<mlir::FloatType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGT, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGT, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGT, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

class GteOpLowering: public ModelicaOpConversion<GteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(GteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		GteOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::sge, lhs, rhs);
		else if (lhsType.isa<RealType>() && rhsType.isa<RealType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGE, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGE, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OGE, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

class LtOpLowering: public ModelicaOpConversion<LtOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LtOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		LtOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::slt, lhs, rhs);
		else if (lhsType.isa<RealType>() && rhsType.isa<RealType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLT, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLT, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLT, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

class LteOpLowering: public ModelicaOpConversion<LteOp>
{
	using ModelicaOpConversion::ModelicaOpConversion;

	mlir::LogicalResult matchAndRewrite(LteOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
	{
		mlir::Location location = op.getLoc();
		LteOp::Adaptor adaptor(operands);

		mlir::Value lhs = adaptor.lhs();
		mlir::Value rhs = adaptor.rhs();

		mlir::Type lhsType = lhs.getType();
		mlir::Type rhsType = rhs.getType();

		mlir::Value result;

		if (lhsType.isa<IntegerType>() && rhsType.isa<IntegerType>())
			result = rewriter.create<mlir::CmpIOp>(location, mlir::CmpIPredicate::sle, lhs, rhs);
		else if (lhsType.isa<RealType>() && rhsType.isa<RealType>())
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLE, lhs, rhs);
		else if (lhsType.isa<IntegerType>() && rhsType.isa<RealType>())
		{
			lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLE, lhs, rhs);
		}
		else if (lhsType.isa<RealType>() && rhsType.isa<IntegerType>())
		{
			rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
			result = rewriter.create<mlir::CmpFOp>(location, mlir::CmpFPredicate::OLE, lhs, rhs);
		}
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");

		rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
		return mlir::success();
	}
};

void ModelicaToLLVMLoweringPass::getDependentDialects(mlir::DialectRegistry &registry) const {
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

	// We need to mark the scf::YieldOp as legal due to a current limitation
	// of MLIR. In fact, YieldOp is used just as a placeholder and would lead
	// to conversion problems when converting the Affine dialect.
	//target.addLegalOp<mlir::AffineYieldOp>();
	//target.addLegalOp<mlir::scf::YieldOp>();

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
	module.dump();

	if (failed(applyPartialConversion(module, target, std::move(patterns))))
	{
		mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
		signalPassFailure();
	}
}

void modelica::populateModelicaToLLVMConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, modelica::TypeConverter& typeConverter)
{
	patterns.insert<AllocaOpLowering, AllocOpLowering, FreeOpLowering, DimOpLowering, LoadOpLowering, StoreOpLowering>(context, typeConverter);
	patterns.insert<IfOpLowering, ForOpLowering, WhileOpLowering>(context, typeConverter);
	patterns.insert<NegateOpLowering, EqOpLowering, NotEqOpLowering, GtOpLowering, GteOpLowering, LtOpLowering, LteOpLowering>(context, typeConverter);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToLLVMLoweringPass(ModelicaToLLVMLoweringOptions options)
{
	return std::make_unique<ModelicaToLLVMLoweringPass>(options);
}
