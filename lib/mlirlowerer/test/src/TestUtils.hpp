#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>

static mlir::ModuleOp wrapFunctionWithModule(mlir::MLIRContext& context, mlir::FuncOp& function)
{
	mlir::OpBuilder builder(&context);
	mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
	module.push_back(function);
	return module;
}

static mlir::FuncOp getFunctionReturningValue(mlir::MLIRContext& context, modelica::Type& returnType, std::function<mlir::Value(modelica::MlirLowerer&)> callback)
{
	modelica::MlirLowerer lowerer(context, false);
	auto& builder = lowerer.getOpBuilder();
	auto functionType = builder.getFunctionType({}, lowerer.lower(returnType));
	mlir::FuncOp function = mlir::FuncOp::create(builder.getUnknownLoc(), "main", functionType);
	auto &entryBlock = *function.addEntryBlock();
	builder.setInsertionPointToStart(&entryBlock);
	builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), callback(lowerer));
	return function;
}
