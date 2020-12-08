#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/utils/IRange.hpp>
#include <set>

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
	for (const auto& name_value : llvm::zip(foo.getArgs(), entryBlock.getArguments())) {
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
		lower(statement);
	}

	return mlir::success();
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const Statement& statement)
{
	return statement.visit([&](auto& obj) { return lower(obj); });
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const AssignmentStatement& statement)
{
	auto destinations = statement.getDestinations();
	auto values = lower<modelica::Expression>(statement.getExpression());
	assert(values.size() == destinations.size() && "Unmatched number of destinations and results");

	Container<std::pair<llvm::StringRef, mlir::Value>> assigned;

	for (auto pair : zip(destinations, values))
	{
		const auto& reference = get<0>(pair)->get<ReferenceAccess>();

		if (!reference.isDummy())
			symbolTable.insert(reference.getName(), get<1>(pair));

		assigned.emplace_back(reference.getName(), get<1>(pair));
	}

	return assigned;
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const IfStatement& statement)
{
	SmallVector<std::map<llvm::StringRef, mlir::Value>, 3> assignments;
	set<llvm::StringRef> vars;
	stack<Block*> blocks;

	for (const auto& conditionalBlock : statement)
	{
		auto condition = lower<modelica::Expression>(conditionalBlock.getCondition())[0];
		auto insertionPoint = builder.saveInsertionPoint();

		Block* trueDest = builder.createBlock(builder.getBlock()->getParent());
		Block* falseDest = builder.createBlock(builder.getBlock()->getParent());

		if (!blocks.empty())
			blocks.pop();

		blocks.push(trueDest);
		blocks.push(falseDest);

		builder.restoreInsertionPoint(insertionPoint);
		builder.create<CondBranchOp>(condition.getLoc(), condition, trueDest, falseDest);

		// First, we create the statements of the current conditional block.
		// While creating them, we keep note of which variables are updated.
		builder.setInsertionPointToStart(trueDest);

		ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
		assignments.emplace_back();

		for (const auto& stmnt : conditionalBlock)
			for (auto& assignment : lower(stmnt))
			{
				assignments[assignments.size() - 1][get<0>(assignment)] = get<1>(assignment);
				vars.emplace(get<0>(assignment));
			}

		// Following code will be inserted in the "else" block. In fact, there is
		// no "else if" block and thus we need to handle them as nested ifs.
		builder.setInsertionPointToStart(falseDest);
	}

	Block* exitBlock = builder.createBlock(builder.getBlock()->getParent());

	assignments.emplace_back();
	SmallVector<mlir::Type, 3> types;

	for (const auto& var : vars)
	{
		auto type = symbolTable.lookup(var).getType();
		types.push_back(type);
	}

	for (size_t i = assignments.size(); i > 0; i--)
	{
		auto& blockAssignments = assignments[i - 1];
		SmallVector<mlir::Value, 3> values;

		for (const auto& var : vars)
			values.emplace_back(blockAssignments.find(var) != blockAssignments.end() ? blockAssignments[var] : symbolTable.lookup(var));

		builder.setInsertionPointToEnd(blocks.top());
		builder.create<BranchOp>(builder.getUnknownLoc(), exitBlock, values);
		blocks.pop();
	}

	exitBlock->addArguments(types);
	builder.setInsertionPointToEnd(exitBlock);

	for (auto pair : zip(vars, exitBlock->getArguments()))
		symbolTable.insert(get<0>(pair), get<1>(pair));

	return {};
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const ForStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return {};
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const WhileStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return {};
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const WhenStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	return {};
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const BreakStatement& statement)
{
	return {};
}

MlirLowerer::Container<std::pair<llvm::StringRef, mlir::Value>> MlirLowerer::lower(const ReturnStatement& statement)
{
	return {};
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

	 if (kind == OperationKind::add)
	{
		auto type = lower(expression.getType());
		SmallVector<mlir::Value, 3> args;

		for (const auto& arg : operation)
		{
			auto tmp = lower<modelica::Expression>(arg);

			//for (auto t : tmp)
				//args.push_back(tmp);
		}

		return { builder.create<AddFOp>(loc(expression.getLocation()), type, args) };
	}

	if (kind == OperationKind::greaterEqual)
	{
		auto lhs = lower<modelica::Expression>(operation[0])[0];
		auto rhs = lower<modelica::Expression>(operation[1])[0];

		return { builder.create<CmpFOp>(loc(expression.getLocation()), CmpFPredicate::OGE, lhs, rhs) };
	}

	return { nullptr };
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Constant>(const Expression& expression)
{
	assert(expression.isA<modelica::Constant>());
	const auto& constant = expression.get<modelica::Constant>();

	auto value = builder.create<ConstantOp>(
			loc(expression.getLocation()),
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
			loc(expression.getLocation()),
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
