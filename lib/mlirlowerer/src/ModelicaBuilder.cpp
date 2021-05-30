#include <modelica/mlirlowerer/ModelicaBuilder.h>

using namespace modelica::codegen;

ModelicaBuilder::ModelicaBuilder(mlir::MLIRContext* context)
		: mlir::OpBuilder(context)
{
}

BooleanType ModelicaBuilder::getBooleanType()
{
	return BooleanType::get(getContext());
}

IntegerType ModelicaBuilder::getIntegerType()
{
	return IntegerType::get(getContext());
}

RealType ModelicaBuilder::getRealType()
{
	return RealType::get(getContext());
}

PointerType ModelicaBuilder::getPointerType(BufferAllocationScope allocationScope, mlir::Type elementType, const PointerType::Shape& shape)
{
	return PointerType::get(getContext(), allocationScope, elementType, shape);
}

OpaquePointerType ModelicaBuilder::getOpaquePointerType()
{
	return OpaquePointerType::get(getContext());
}

StructType ModelicaBuilder::getStructType(llvm::ArrayRef<mlir::Type> types)
{
	return StructType::get(getContext(), types);
}

mlir::IntegerAttr ModelicaBuilder::getIndexAttribute(long value)
{
	return getIndexAttr(value);
}

BooleanAttribute ModelicaBuilder::getBooleanAttribute(bool value)
{
	return BooleanAttribute::get(getBooleanType(), value);
}

IntegerAttribute ModelicaBuilder::getIntegerAttribute(long value)
{
	return IntegerAttribute::get(getIntegerType(), value);
}

RealAttribute ModelicaBuilder::getRealAttribute(double value)
{
	return RealAttribute::get(getRealType(), value);
}

mlir::Attribute ModelicaBuilder::getZeroAttribute(mlir::Type type)
{
	if (type.isa<BooleanType>())
		return getBooleanAttribute(false);

	if (type.isa<IntegerType>())
		return getIntegerAttribute(0);

	if (type.isa<RealType>())
		return getRealAttribute(0);

	return getZeroAttr(type);
}

InverseFunctionsAttribute ModelicaBuilder::getInverseFunctionsAttribute(InverseFunctionsAttribute::Map map)
{
	return InverseFunctionsAttribute::get(getContext(), map);
}

DerivativeAttribute ModelicaBuilder::getDerivativeAttribute(llvm::StringRef name, unsigned int order)
{
	return DerivativeAttribute::get(getContext(), name, order);
}
