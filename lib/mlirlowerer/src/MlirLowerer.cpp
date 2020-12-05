#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/mlirlowerer/ConstantOpOld.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ReturnOp.hpp>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

MlirLowerer::MlirLowerer(mlir::MLIRContext& context) : builder(&context)
{
}

FuncOp MlirLowerer::lower(const ClassContainer& cls)
{
	return cls.visit([&](auto& obj) { return lower(obj); });
}

FuncOp MlirLowerer::lower(const Class& cls)
{
	return nullptr;
}


FuncOp MlirLowerer::lower(const Function& foo)
{
	// Create a scope in the symbol table to hold variable declarations.
	ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

	auto location = loc(foo.getSourcePosition());

	SmallVector<mlir::Type, 3> argTypes;
	SmallVector<mlir::Type, 3> returnTypes;

	for (const auto& member : foo.getArgs())
		argTypes.push_back(lower(member->getType()));

	for (const auto& member : foo.getResults())
			returnTypes.push_back(lower(member->getType()));

	auto functionType = builder.getFunctionType(argTypes, returnTypes);
	auto function = mlir::FuncOp::create(location, foo.getName(), functionType);

	// Start the body of the function.
	// In MLIR the entry block of the function is special: it must have the same
	// argument list as the function itself.
	auto &entryBlock = *function.addEntryBlock();

	// Declare all the function arguments in the symbol table
	for (const auto &name_value : llvm::zip(foo.getArgs(), entryBlock.getArguments())) {
		if (failed(declare(get<0>(name_value)->getName(), get<1>(name_value))))
			return nullptr;
	}

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Emit the body of the function
	if (mlir::failed(lower(foo.getAlgorithms()[0]))) {
		function.erase();
		return nullptr;
	}

	SmallVector<mlir::Value, 3> results;

	for (const auto& member : foo.getResults())
		results.push_back(symbolTable.lookup(member->getName()));

	if (!returnTypes.empty())
		builder.create<ReturnOp>(loc(SourcePosition("-", 0, 0)), returnTypes, results);

	return function;
}

mlir::Location MlirLowerer::loc(SourcePosition location) {
	return builder.getFileLineColLoc(builder.getIdentifier(*location.file),
																	 location.line,
																	 location.column);
}

LogicalResult MlirLowerer::declare(StringRef var, Value value) {
	if (symbolTable.count(var) != 0)
		return failure();

	symbolTable.insert(var, value);
	return success();
}

mlir::Type MlirLowerer::lower(const Type& type)
{
	return type.visit([&](auto& obj) { return lower(obj); });
}

mlir::Type MlirLowerer::lower(const BuiltInType& type)
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

mlir::Type MlirLowerer::lower(const UserDefinedType& type)
{
	SmallVector<mlir::Type, 3> types;

	for (auto& subType : type)
		types.push_back(lower(subType));

	return builder.getTupleType(move(types));
}

mlir::LogicalResult MlirLowerer::lower(const Algorithm& algorithm)
{
	for (const auto& statement : algorithm) {
		if (failed(lower(statement)))
			return failure();
	}

	return mlir::success();
}

mlir::LogicalResult MlirLowerer::lower(const Statement& statement)
{
	if (failed(statement.visit([&](auto& obj) { return lower(obj); })))
		return failure();

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const AssignmentStatement& statement)
{
	Value value = lower(statement.getExpression());

	// Register the value in the symbol table.
	if (!statement.getDestinations()[0]->isA<ReferenceAccess>())
		return success();

	if (failed(declare(statement.getDestinations()[0]->get<ReferenceAccess>().getName(), value)))
		return failure();

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const IfStatement& statement)
{
	ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const ForStatement& statement)
{
	ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const WhileStatement& statement)
{
	ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const WhenStatement& statement)
{
	ScopedHashTableScope<StringRef, Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const BreakStatement& statement)
{
	return success();
}

mlir::LogicalResult MlirLowerer::lower(const ReturnStatement& statement)
{
	return success();
}

mlir::Value MlirLowerer::lower(const Expression& expression)
{
	return expression.visit([&](auto& obj) { return lower(obj); });
}

mlir::Value MlirLowerer::lower(const modelica::Operation& operation)
{
	return nullptr;
}

mlir::Value MlirLowerer::lower(const Constant& constant)
{
	return builder.create<ConstantOp>(
			loc(SourcePosition("-", 0, 0)),
			constantToType(constant),
			constant.visit([&](const auto& obj) { return getAttribute(obj); })
			);
	/*
	return builder.create<ConstantOpOld>(
			loc(SourcePosition("-", 0, 0)),
			constantToType(constant),
			constant.visit([&](const auto& obj) { return getAttribute(obj); })
			);
			*/
}

mlir::Value MlirLowerer::lower(const ReferenceAccess& reference)
{
	return nullptr;
}

mlir::Value MlirLowerer::lower(const Call& call)
{
	return nullptr;
}

mlir::Value MlirLowerer::lower(const Tuple& tuple)
{
	return nullptr;
}
