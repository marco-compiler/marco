#pragma once

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>

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
