#include "marco/Codegen/ArrayDescriptor.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  ArrayDescriptor::ArrayDescriptor(mlir::LLVMTypeConverter* typeConverter, mlir::Value value)
      : typeConverter(typeConverter),
        value(value),
        descriptorType(value.getType())
  {
    assert(value != nullptr && "Value cannot be null");
    assert(descriptorType.isa<mlir::LLVM::LLVMStructType>() && "Expected LLVM struct type");
  }

  ArrayDescriptor ArrayDescriptor::undef(
      mlir::OpBuilder& builder, mlir::LLVMTypeConverter* typeConverter, mlir::Location location, mlir::Type descriptorType)
  {
    mlir::Value descriptor = builder.create<mlir::LLVM::UndefOp>(location, descriptorType);
    return ArrayDescriptor(typeConverter, descriptor);
  }

  mlir::Value ArrayDescriptor::operator*()
  {
    return value;
  }

  mlir::Value ArrayDescriptor::getPtr(mlir::OpBuilder& builder, mlir::Location location)
  {
    mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[0];
    return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(0));
  }

  void ArrayDescriptor::setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr)
  {
    value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, ptr, builder.getIndexArrayAttr(0));
  }

  mlir::Value ArrayDescriptor::getRank(mlir::OpBuilder& builder, mlir::Location location)
  {
    mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
    return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(1));
  }

  void ArrayDescriptor::setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank)
  {
    value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, rank, builder.getIndexArrayAttr(1));
  }

  mlir::Value ArrayDescriptor::getSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension)
  {
    mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[2];
    type = type.cast<mlir::LLVM::LLVMArrayType>().getElementType();
    return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr({ 2, dimension }));
  }

  mlir::Value ArrayDescriptor::getSize(mlir::OpBuilder& builder, mlir::Location location, mlir::Value dimension)
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

  void ArrayDescriptor::setSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension, mlir::Value size)
  {
    value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, size, builder.getIndexArrayAttr({ 2, dimension }));
  }

  mlir::Value ArrayDescriptor::computeSize(mlir::OpBuilder& builder, mlir::Location loc)
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

  mlir::Type ArrayDescriptor::getRankType() const
  {
    auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();
    mlir::Type rankType = body[1];
    assert(rankType.isa<mlir::IntegerType>() && "The rank must have integer type");
    return rankType;
  }

  mlir::Type ArrayDescriptor::getSizesContainerType() const
  {
    auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();
    assert(body.size() >= 3);
    auto sizesArrayType = body[2];
    assert(sizesArrayType.isa<mlir::LLVM::LLVMArrayType>() && "The sizes of the array must be contained into an array");
    return sizesArrayType;
  }

  mlir::Type ArrayDescriptor::getSizeType() const
  {
    auto body = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody();

    if (body.size() == 2) {
      return typeConverter->convertType(mlir::IndexType::get(value.getContext()));
    }

    mlir::Type sizesContainerType = getSizesContainerType();
    mlir::Type sizeType = sizesContainerType.cast<mlir::LLVM::LLVMArrayType>().getElementType();
    assert(sizeType.isa<mlir::IntegerType>() && "Each size of the array must have integer type");
    return sizeType;
  }

  UnsizedArrayDescriptor::UnsizedArrayDescriptor(mlir::Value value)
      : value(value),
        descriptorType(value.getType())
  {
    assert(value != nullptr && "Value cannot be null");
    assert(descriptorType.isa<mlir::LLVM::LLVMStructType>() && "Expected LLVM struct type");
  }

  UnsizedArrayDescriptor UnsizedArrayDescriptor::undef(
      mlir::OpBuilder& builder, mlir::Location location, mlir::Type descriptorType)
  {
    mlir::Value descriptor = builder.create<mlir::LLVM::UndefOp>(location, descriptorType);
    return UnsizedArrayDescriptor(descriptor);
  }

  mlir::Value UnsizedArrayDescriptor::operator*()
  {
    return value;
  }

  mlir::Value UnsizedArrayDescriptor::getRank(mlir::OpBuilder& builder, mlir::Location location)
  {
    mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[0];
    return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(0));
  }

  void UnsizedArrayDescriptor::setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank)
  {
    value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, rank, builder.getIndexArrayAttr(0));
  }

  mlir::Value UnsizedArrayDescriptor::getPtr(mlir::OpBuilder& builder, mlir::Location location)
  {
    mlir::Type type = descriptorType.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
    return builder.create<mlir::LLVM::ExtractValueOp>(location, type, value, builder.getIndexArrayAttr(1));
  }

  void UnsizedArrayDescriptor::setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr)
  {
    value = builder.create<mlir::LLVM::InsertValueOp>(location, descriptorType, value, ptr, builder.getIndexArrayAttr(1));
  }
}
