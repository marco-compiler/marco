#include <marco/mlirlowerer/passes/TypeConverter.h>

using namespace marco::codegen;
using namespace modelica;

TypeConverter::TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth)
		: mlir::LLVMTypeConverter(context, options), bitWidth(bitWidth)
{
	addConversion([&](BooleanType type) { return convertBooleanType(type); });
	addConversion([&](IntegerType type) { return convertIntegerType(type); });
	addConversion([&](RealType type) { return convertRealType(type); });
	addConversion([&](ArrayType type) { return convertArrayType(type); });
	addConversion([&](UnsizedArrayType type) { return convertUnsizedArrayType(type); });
	addConversion([&](OpaquePointerType type) { return convertOpaquePointerType(type); });
	addConversion([&](StructType type) { return convertStructType(type); });

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

				if (!inputs[0].getType().isa<ArrayType>() &&
				    !inputs[0].getType().isa<UnsizedArrayType>() &&
						!inputs[0].getType().isa<StructType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addTargetMaterialization(
			[&](mlir::OpBuilder &builder, mlir::LLVM::LLVMPointerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<OpaquePointerType>())
					return llvm::None;

				return builder.create<mlir::LLVM::BitcastOp>(loc, resultType, inputs[0]).getResult();
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

				if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != this->bitWidth)
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, RealType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::FloatType>() || inputs[0].getType().getIntOrFloatBitWidth() != this->bitWidth)
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, ArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				auto isZeroDimensionalPointer = [](mlir::Type type) -> bool {
					if (auto structType = type.dyn_cast<mlir::LLVM::LLVMStructType>())
					{
						if (auto types = structType.getBody(); types.size() == 2)
						{
							if (types[0].isa<mlir::LLVM::LLVMPointerType>() &&
									types[1].isa<mlir::IntegerType>())
								return true;
						}
					}

					return false;
				};

				auto isMultiDimensionalPointer = [](mlir::Type type) -> bool {
					if (auto structType = type.dyn_cast<mlir::LLVM::LLVMStructType>())
					{
						if (auto types = structType.getBody(); types.size() == 3)
						{
							if (types[0].isa<mlir::LLVM::LLVMPointerType>() &&
									types[1].isa<mlir::IntegerType>() &&
									types[2].isa<mlir::LLVM::LLVMArrayType>() &&
									types[2].cast<mlir::LLVM::LLVMArrayType>().getElementType().isa<mlir::IntegerType>())
								return true;
						}
					}

					return false;
				};

				if (isZeroDimensionalPointer(inputs[0].getType()) || isMultiDimensionalPointer(inputs[0].getType()))
					return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);

				return llvm::None;
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, UnsizedArrayType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (auto structType = inputs[0].getType().dyn_cast<mlir::LLVM::LLVMStructType>())
					if (auto types = structType.getBody(); types.size() == 2)
						if (types[0].isa<mlir::IntegerType>() &&
								types[1].isa<mlir::LLVM::LLVMPointerType>())
							return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);

				return llvm::None;
			});

	addSourceMaterialization(
			[&](mlir::OpBuilder &builder, StructType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value>
			{
				if (inputs.size() != 1)
					return llvm::None;

				if (!inputs[0].getType().isa<mlir::LLVM::LLVMStructType>())
					return llvm::None;

				return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
			});
}

mlir::Type TypeConverter::convertBooleanType(BooleanType type)
{
	return mlir::IntegerType::get(&getContext(), 1);
}

mlir::Type TypeConverter::convertIntegerType(IntegerType type)
{
	return mlir::IntegerType::get(&getContext(), bitWidth);
}

mlir::Type TypeConverter::convertRealType(RealType type)
{
	if (bitWidth == 16)
		return convertType(mlir::Float16Type::get(&getContext()));

	if (bitWidth == 32)
		return convertType(mlir::Float32Type::get(&getContext()));

	if (bitWidth == 64)
		return convertType(mlir::Float64Type::get(&getContext()));

	mlir::emitError(mlir::UnknownLoc::get(&getContext())) << "Unsupported type: !modelica.real<" << bitWidth << ">";
	return {};
}

mlir::Type TypeConverter::convertArrayType(ArrayType type)
{
	auto types = getArrayDescriptorFields(type);
	return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), types);
}

mlir::Type TypeConverter::convertUnsizedArrayType(UnsizedArrayType type)
{
	auto types = getUnsizedArrayDescriptorFields(type);
	return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), types);
}

mlir::Type TypeConverter::convertOpaquePointerType(OpaquePointerType type)
{
	return mlir::LLVM::LLVMPointerType::get(convertType(mlir::IntegerType::get(type.getContext(), 8)));
}

mlir::Type TypeConverter::convertStructType(StructType type)
{
	llvm::SmallVector<mlir::Type, 3> subtypes;

	for (const auto& subtype : type.getElementTypes())
		subtypes.push_back(convertType(subtype));

	return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), subtypes);
}

llvm::SmallVector<mlir::Type, 3> TypeConverter::getArrayDescriptorFields(ArrayType type)
{
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

llvm::SmallVector<mlir::Type, 3> TypeConverter::getUnsizedArrayDescriptorFields(UnsizedArrayType type)
{
	auto indexType = getIndexType();
	auto voidPtr = mlir::LLVM::LLVMPointerType::get(convertType(mlir::IntegerType::get(type.getContext(), 8)));

	llvm::SmallVector<mlir::Type, 3> results = { indexType, voidPtr };
	return results;
}
