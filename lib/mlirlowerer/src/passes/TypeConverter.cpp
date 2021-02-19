#include <modelica/mlirlowerer/passes/TypeConverter.h>

using namespace modelica;

mlir::Type TypeConverter::indexType()
{
	return convertType(mlir::IndexType::get(&getContext()));
}

mlir::Type TypeConverter::voidPtrType()
{
	return mlir::LLVM::LLVMPointerType::get(convertType(IntegerType::get(&getContext(), 8)));
}

mlir::Type TypeConverter::convertBooleanType(BooleanType type)
{
	return convertType(mlir::IntegerType::get(&getContext(), 1));
}

mlir::Type TypeConverter::convertIntegerType(IntegerType type)
{
	return convertType(mlir::IntegerType::get(&getContext(), type.getBitWidth()));
}

mlir::Type TypeConverter::convertRealType(RealType type)
{
	unsigned int bitWidth = type.getBitWidth();

	if (bitWidth == 16)
		return convertType(mlir::Float16Type::get(&getContext()));

	if (bitWidth == 32)
		return convertType(mlir::Float32Type::get(&getContext()));

	if (bitWidth == 64)
		return convertType(mlir::Float64Type::get(&getContext()));

	mlir::emitError(mlir::UnknownLoc::get(&getContext())) << "Unsupported type: !modelica.real<" << bitWidth << ">";
	return {};
}

mlir::Type TypeConverter::convertPointerType(PointerType type)
{
	auto types = getPointerDescriptorFields(type);
	return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), types);
}

llvm::SmallVector<mlir::Type, 3> TypeConverter::getPointerDescriptorFields(PointerType type) {
	mlir::Type elementType = type.getElementType();
	elementType = convertType(elementType);

	auto ptrType = mlir::LLVM::LLVMPointerType::get(elementType, 0);
	auto indexType = getIndexType();
	llvm::SmallVector<mlir::Type, 3> results = { ptrType, indexType };

	auto rank = type.getRank();

	if (rank == 0)
		return results;

	results.insert(results.end(), 1, mlir::LLVM::LLVMArrayType::get(indexType, rank));
	return results;
}
