#include <modelica/mlirlowerer/ModelicaBuilder.h>

using namespace modelica;

BooleanType ModelicaBuilder::getBooleanType()
{
	return BooleanType::get(getContext());
}

IntegerType ModelicaBuilder::getIntegerType()
{
	return IntegerType::get(getContext(), 64);
}

RealType ModelicaBuilder::getRealType()
{
	return RealType::get(getContext(), 64);
}

PointerType ModelicaBuilder::getPointerType(bool heap, mlir::Type elementType, const PointerType::Shape& shape)
{
	return PointerType::get(getContext(), heap, elementType, shape);
}

mlir::IntegerAttr ModelicaBuilder::getBooleanAttribute(bool value)
{
	return getBoolAttr(value);
}

mlir::IntegerAttr ModelicaBuilder::getIndexAttribute(long value)
{
	return getIndexAttr(value);
}

mlir::IntegerAttr ModelicaBuilder::getIntegerAttribute(long value)
{
	return getI64IntegerAttr(value);
}

mlir::FloatAttr ModelicaBuilder::getRealAttribute(double value)
{
	return getF64FloatAttr(value);
}
