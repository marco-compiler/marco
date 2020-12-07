#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/utils/IRange.hpp>

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
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

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
		symbolTable.insert(get<0>(name_value)->getName(),  get<1>(name_value));
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

	std::vector<mlir::Value> results;

	for (const auto& member : foo.getResults())
		results.push_back(symbolTable.lookup(member->getName()));

	if (!returnTypes.empty())
		builder.create<ReturnOp>(builder.getUnknownLoc(), results);

	return function;
}

mlir::Location MlirLowerer::loc(SourcePosition location)
{
	return builder.getFileLineColLoc(builder.getIdentifier(*location.file),
																	 location.line,
																	 location.column);
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
	auto destinations = statement.getDestinations();
	auto values = lower<modelica::Expression>(statement.getExpression());
	assert(values.size() == destinations.size() && "Unmatched number of destinations and results");

	for (auto pair : zip(destinations, values))
		symbolTable.insert(get<0>(pair)->get<ReferenceAccess>().getName(), get<1>(pair));

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const IfStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const ForStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const WhileStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return success();
}

mlir::LogicalResult MlirLowerer::lower(const WhenStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

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

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<Expression>(const Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return lower<deconst>(expression);
	});
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Operation>(const Expression& expression)
{
	assert(expression.isA<modelica::Operation>());
	const auto& operation = expression.get<modelica::Operation>();
	auto kind = operation.getKind();

	if (kind == OperationKind::negate)
	{
		return { nullptr };
	}
	else if (kind == OperationKind::add)
	{
		auto type = lower(expression.getType());
		SmallVector<mlir::Value, 3> args;

		for (const auto& arg : operation)
		{
			auto tmp = lower<modelica::Expression>(arg);

			//for (auto t : tmp)
				//args.push_back(tmp);
		}

		return { builder.create<AddFOp>(builder.getUnknownLoc(), type, args) };
	}

	return { nullptr };
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Constant>(const Expression& expression)
{
	assert(expression.isA<modelica::Constant>());
	const auto& constant = expression.get<modelica::Constant>();

	auto value = builder.create<ConstantOp>(
			builder.getUnknownLoc(),
			constantToType(constant),
			constant.visit([&](const auto& obj) { return getAttribute(obj); }));

	return { value };
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::ReferenceAccess>(const Expression& expression)
{
	assert(expression.isA<modelica::ReferenceAccess>());
	const auto& reference = expression.get<modelica::ReferenceAccess>();
	return { symbolTable.lookup(reference.getName()) };
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Call>(const Expression& expression)
{
	assert(expression.isA<modelica::Call>());
	const auto& call = expression.get<modelica::Call>();
	const auto& function = call.getFunction();

	SmallVector<mlir::Value, 3> args;

	for (const auto& arg : call)
		for (auto& val : lower<modelica::Expression>(arg))
			args.push_back(val);

	auto op = builder.create<CallOp>(
			builder.getUnknownLoc(),
			function.get<ReferenceAccess>().getName(),
			lower(function.getType()),
			args);

	Container<mlir::Value> results;

	for (auto result : op.getResults())
		results.push_back(result);

	return results;
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Tuple>(const Expression& expression)
{
	assert(expression.isA<modelica::Tuple>());
	const auto& tuple = expression.get<modelica::Tuple>();
	Container<mlir::Value> result;

	for (auto& exp : tuple)
	{
		auto values = lower<modelica::Expression>(expression);

		// The only way to have multiple returns is to call a function, but this
		// is forbidden in a tuple declaration. In fact, a tuple is just a
		// container of references.
		assert(values.size() == 1);
		result.push_back(values[0]);
	}

	return result;
}
