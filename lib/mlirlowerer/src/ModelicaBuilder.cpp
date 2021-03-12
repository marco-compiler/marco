#include <modelica/mlirlowerer/ModelicaBuilder.h>

using namespace modelica;

ModelicaBuilder::ModelicaBuilder(mlir::MLIRContext* context, unsigned int bitWidth)
		: mlir::OpBuilder(context),
			bitWidth(bitWidth)
{
}

BooleanType ModelicaBuilder::getBooleanType()
{
	return BooleanType::get(getContext());
}

IntegerType ModelicaBuilder::getIntegerType()
{
	return IntegerType::get(getContext(), bitWidth);
}

RealType ModelicaBuilder::getRealType()
{
	return RealType::get(getContext(), bitWidth);
}

PointerType ModelicaBuilder::getPointerType(bool heap, mlir::Type elementType, const PointerType::Shape& shape)
{
	return PointerType::get(getContext(), heap, elementType, shape);
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