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
	if (bitWidth == 8)
		return getI8IntegerAttr(value);

	if (bitWidth == 16)
		return getI16IntegerAttr(value);

	if (bitWidth == 32)
		return getI32IntegerAttr(value);

	if (bitWidth == 64)
		return getI64IntegerAttr(value);

	assert(false && "Unsupported bit width");
}

mlir::FloatAttr ModelicaBuilder::getRealAttribute(double value)
{
	if (bitWidth == 16)
		return getF16FloatAttr(value);

	if (bitWidth == 32)
		return getF32FloatAttr(value);

	if (bitWidth == 64)
		return getF64FloatAttr(value);

	assert(false && "Unsupported bit width");
}
