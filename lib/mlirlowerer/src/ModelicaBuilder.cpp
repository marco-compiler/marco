#include <modelica/mlirlowerer/ModelicaBuilder.h>

using namespace modelica;

BooleanType ModelicaBuilder::getBooleanType()
{
	return BooleanType::get(getContext());
}

IntegerType ModelicaBuilder::getIntegerType()
{
	return IntegerType::get(getContext(), 32);
}

RealType ModelicaBuilder::getRealType()
{
	return RealType::get(getContext(), 32);
}

PointerType ModelicaBuilder::getPointerType(mlir::Type elementType, const PointerType::Shape& shape, mlir::AffineMapAttr map)
{
	return PointerType::get(getContext(), elementType, shape, map);
}

/*
IndexAttribute ModelicaBuilder::getIndexAttribute(long value)
{
	return IndexAttribute::get(getContext(), value);
}
*/
