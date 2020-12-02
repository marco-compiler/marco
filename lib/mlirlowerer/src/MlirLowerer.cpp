#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

MlirLowerer::MlirLowerer(mlir::MLIRContext& context) : builder(&context)
{
}

FuncOp MlirLowerer::lower(ClassContainer cls)
{
	return cls.visit([&](auto& obj) { return lower(obj); });
}

FuncOp MlirLowerer::lower(Class cls)
{
	return nullptr;
}

FuncOp MlirLowerer::lower(Function function)
{
	auto location = loc(function.getSourcePosition());

	SmallVector<mlir::Type, 3> argTypes;
	SmallVector<mlir::Type, 3> returnTypes;

	for (const auto& member : function.getMembers())
	{
		if (member.isInput())
			argTypes.push_back(lower(member.getType()));
		else if (member.isOutput())
			returnTypes.push_back(lower(member.getType()));
	}

	auto functionType = builder.getFunctionType(argTypes, returnTypes);
	auto result = mlir::FuncOp::create(location, function.getName(), functionType);

	// Start the body of the function.
	auto &entryBlock = *result.addEntryBlock();

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Emit the body of the function.

	/*
	// Implicitly return void if no return statement was emitted.
	ReturnOp returnOp;

	if (!entryBlock.empty())
		returnOp = dyn_cast<ReturnOp>(entryBlock.back());

	if (!returnOp)
	{
		builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));

	} else if (returnOp.hasOperand()) {
		// Otherwise, if this return operation has an operand then add a result to
		// the function.
		function.setType(builder.getFunctionType(function.getType().getInputs(),
																						 getType(VarType{})));
	}
*/
	return result;
}

mlir::Location MlirLowerer::loc(SourcePosition location) {
	return builder.getFileLineColLoc(builder.getIdentifier(*location.file),
																	 location.line,
																	 location.column);
}

mlir::Type MlirLowerer::lower(Type type)
{
	return type.visit([&](auto& obj) { return lower(obj); });
}

mlir::Type MlirLowerer::lower(BuiltInType type)
{
	switch (type)
	{
		case BuiltInType::None:
			return builder.getNoneType();
		case BuiltInType::Integer:
			return builder.getI32Type();
		case BuiltInType::Float:
			return builder.getF32Type();
		case BuiltInType::Boolean:
			return builder.getI1Type();
		default:
			assert(false && "Unexpected type");
	}
}

mlir::Type MlirLowerer::lower(UserDefinedType type)
{
	SmallVector<mlir::Type, 3> types;

	for (const auto& subType : type)
		types.push_back(lower(subType));

	return builder.getTupleType(move(types));
}

void MlirLowerer::lower(Algorithm algorithm)
{
	ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);

	/*
	for (auto& statement : algorithm) {
		// Specific handling for variable declarations, return statement, and
		// print. These can only appear in block list and not in nested
		// expressions.
		if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
			if (!mlirGen(*vardecl))
				return mlir::failure();
			continue;
		}
		if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
			return mlirGen(*ret);
		if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
			if (mlir::failed(mlirGen(*print)))
				return mlir::success();
			continue;
		}

		// Generic expression dispatch codegen.
		if (!mlirGen(*expr))
			return mlir::failure();
	}

	return mlir::success();
*/
}

void MlirLowerer::lower(Statement statement)
{

}

void MlirLowerer::lower(AssignmentStatement statement)
{

}

void MlirLowerer::lower(IfStatement statement)
{

}

void MlirLowerer::lower(ForStatement statement)
{

}

void MlirLowerer::lower(WhileStatement statement)
{

}

void MlirLowerer::lower(WhenStatement statement)
{

}
