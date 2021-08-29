#include <marco/mlirlowerer/dialects/ida/IdaBuilder.h>

using namespace marco::codegen::ida;

IdaBuilder::IdaBuilder(mlir::MLIRContext* context)
		: mlir::OpBuilder(context)
{
}

BooleanType IdaBuilder::getBooleanType()
{
	return BooleanType::get(getContext());
}

IntegerType IdaBuilder::getIntegerType()
{
	return IntegerType::get(getContext());
}

RealType IdaBuilder::getRealType()
{
	return RealType::get(getContext());
}

BooleanAttribute IdaBuilder::getBooleanAttribute(bool value)
{
	return BooleanAttribute::get(getBooleanType(), value);
}

IntegerAttribute IdaBuilder::getIntegerAttribute(long value)
{
	return IntegerAttribute::get(getIntegerType(), value);
}

RealAttribute IdaBuilder::getRealAttribute(double value)
{
	return RealAttribute::get(getRealType(), value);
}
