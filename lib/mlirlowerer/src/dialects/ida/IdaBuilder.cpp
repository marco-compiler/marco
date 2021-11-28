#include <marco/mlirlowerer/dialects/ida/IdaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

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

OpaquePointerType IdaBuilder::getOpaquePointerType()
{
	return OpaquePointerType::get(getContext());
}

IntegerPointerType IdaBuilder::getIntegerPointerType()
{
	return IntegerPointerType::get(getContext());
}

RealPointerType IdaBuilder::getRealPointerType()
{
	return RealPointerType::get(getContext());
}

mlir::Type IdaBuilder::getResidualFunctionType()
{
	mlir::Type resultType = modelica::RealType::get(getContext());
	llvm::SmallVector<mlir::Type, 4> argTypes = {
		modelica::RealType::get(getContext()),
		getRealPointerType(),
		getRealPointerType(),
		getIntegerPointerType()
	};

	return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(resultType, argTypes));
}

mlir::Type IdaBuilder::getJacobianFunctionType()
{
	mlir::Type resultType = modelica::RealType::get(getContext());
	llvm::SmallVector<mlir::Type, 6> argTypes = {
		modelica::RealType::get(getContext()),
		getRealPointerType(),
		getRealPointerType(),
		getIntegerPointerType(),
		modelica::RealType::get(getContext()),
		modelica::IntegerType::get(getContext()),
	};

	return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(resultType, argTypes));
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
