#include <marco/mlirlowerer/dialects/ida/IdaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

using namespace marco::codegen::ida;

IdaBuilder::IdaBuilder(mlir::MLIRContext* context)
		: modelica::ModelicaBuilder(context)
{
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

llvm::SmallVector<mlir::Type, 4> IdaBuilder::getResidualArgTypes()
{
	return {
		getRealType(),
		getRealPointerType(),
		getRealPointerType(),
		getIntegerPointerType()
	};
}

mlir::Type IdaBuilder::getResidualFunctionType()
{
	return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(getRealType(), getResidualArgTypes()));
}

llvm::SmallVector<mlir::Type, 6> IdaBuilder::getJacobianArgTypes()
{
	return {
		getRealType(),
		getRealPointerType(),
		getRealPointerType(),
		getIntegerPointerType(),
		getRealType(),
		getIntegerType(),
	};
}

mlir::Type IdaBuilder::getJacobianFunctionType()
{
	return mlir::LLVM::LLVMPointerType::get(mlir::LLVM::LLVMFunctionType::get(getRealType(), getJacobianArgTypes()));
}
