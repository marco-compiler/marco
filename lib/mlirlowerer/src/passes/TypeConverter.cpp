#include <modelica/mlirlowerer/passes/TypeConverter.h>

using namespace modelica;

TypeConverter::TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options) : mlir::LLVMTypeConverter(context, options)
{
	addConversion([&](BooleanType type) { return convertBooleanType(type); });
	addConversion([&](IntegerType type) { return convertIntegerType(type); });
	addConversion([&](RealType type) { return convertRealType(type); });
	addConversion([&](PointerType type) { return convertPointerType(type); });
	addConversion([&](RecordType type) { return convertRecordType(type); });

	addTargetMaterialization(
	[&](mlir::OpBuilder &builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
	    {
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<BooleanType>() && !inputs[0].getType().isa<IntegerType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addTargetMaterialization(
			[&](mlir::OpBuilder &builder, mlir::FloatType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			    {
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<RealType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addTargetMaterialization(
			[&](mlir::OpBuilder &builder, mlir::LLVM::LLVMStructType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<PointerType>() && !inputs[0].getType().isa<RecordType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != 1)
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != resultType.getBitWidth())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, RealType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::FloatType>() || inputs[0].getType().getIntOrFloatBitWidth() != resultType.getBitWidth())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, PointerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::LLVM::LLVMStructType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, RecordType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::LLVM::LLVMStructType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});
}

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
	return mlir::IntegerType::get(&getContext(), 1);
}

mlir::Type TypeConverter::convertIntegerType(IntegerType type)
{
	return mlir::IntegerType::get(&getContext(), type.getBitWidth());
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
	return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), types);
}

mlir::Type TypeConverter::convertRecordType(RecordType type)
{
	llvm::SmallVector<mlir::Type, 3> subtypes;

	for (const auto& subtype : type.getElementTypes())
		subtypes.push_back(convertType(subtype));

	return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), subtypes);
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
