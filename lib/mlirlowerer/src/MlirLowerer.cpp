#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/utils/IRange.hpp>
#include <set>

#include <mlir/IR/PatternMatch.h>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

void modelica::registerDialects()
{
	mlir::registerDialect<ModelicaDialect>();
	mlir::registerDialect<StandardOpsDialect>();
	mlir::registerDialect<scf::SCFDialect>();
	mlir::registerDialect<LLVM::LLVMDialect>();
}

mlir::LogicalResult modelica::convertToLLVMDialect(mlir::MLIRContext* context, mlir::ModuleOp module)
{
	mlir::PassManager passManager(context);
	passManager.addPass(std::make_unique<LLVMLoweringPass>());
	return passManager.run(module);
}

MlirLowerer::MlirLowerer(mlir::MLIRContext& context) : builder(&context)
{
	// Check that the required dialects have been previously registered
	assert(context.getRegisteredDialect<ModelicaDialect>() != nullptr);
	assert(context.getRegisteredDialect<StandardOpsDialect>() != nullptr);
	assert(context.getRegisteredDialect<scf::SCFDialect>() != nullptr);
	assert(context.getRegisteredDialect<LLVM::LLVMDialect>() != nullptr);
}

mlir::ModuleOp MlirLowerer::lower(llvm::ArrayRef<const modelica::ClassContainer> classes)
{
	mlir::ModuleOp module = ModuleOp::create(builder.getUnknownLoc());

	for (const auto& cls : classes)
	{
		auto* op = cls.visit([&](auto& obj) -> mlir::Operation* { return lower(obj); });
		module.push_back(op);
	}

	return module;
}

mlir::Operation* MlirLowerer::lower(const modelica::Class& cls)
{
	return nullptr;
}

mlir::FuncOp MlirLowerer::lower(const modelica::Function& foo)
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

	// Initialize members
	for (const auto& member : foo.getMembers())
		if (!member.isInput())
			lower(member);

	// Emit the body of the function
	lower(foo.getAlgorithms()[0]);

	// Return statement
	std::vector<mlir::Value> results;

	for (const auto& member : foo.getResults())
	{
		auto reference = symbolTable.lookup(member->getName());
		auto value = builder.create<LoadOp>(builder.getUnknownLoc(), reference);
		results.push_back(value);
	}

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

mlir::MemRefType MlirLowerer::lower(const modelica::Type& type)
{
	auto visitor = [&](auto& obj)
	{
		auto baseType = lower(obj);

		if (!type.isScalar())
		{
			const auto& dimensions = type.getDimensions();
			SmallVector<long, 3> shape(dimensions.begin(), dimensions.end());
			return MemRefType::get(shape, baseType);
		}

		return MemRefType::get({}, baseType);
	};

	return type.visit(visitor);
}

mlir::Type MlirLowerer::lower(const modelica::BuiltInType& type)
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

	return builder.getNoneType();
}

mlir::Type MlirLowerer::lower(const modelica::UserDefinedType& type)
{
	SmallVector<mlir::Type, 3> types;

	for (auto& subType : type)
		types.push_back(lower(subType));

	return builder.getTupleType(move(types));
}

void MlirLowerer::lower(const modelica::Member& member)
{
	auto type = lower(member.getType());
	auto var = builder.create<AllocaOp>(builder.getUnknownLoc(), type);
	symbolTable.insert(member.getName(), var);

	if (member.hasInitializer())
	{
		auto values = lower<modelica::Expression>(member.getInitializer());
		assert(!values.empty());
		builder.create<StoreOp>(builder.getUnknownLoc(), values[0], var);
	}
}

void MlirLowerer::lower(const modelica::Algorithm& algorithm)
{
	for (const auto& statement : algorithm) {
		lower(statement);
	}
}

void MlirLowerer::lower(const modelica::Statement& statement)
{
	statement.visit([&](auto& obj) { lower(obj); });
}

void MlirLowerer::lower(const modelica::AssignmentStatement& statement)
{
	auto destinations = statement.getDestinations();
	auto values = lower<modelica::Expression>(statement.getExpression());
	assert(values.size() == destinations.size() && "Unmatched number of destinations and results");

	for (auto pair : zip(destinations, values))
	{
		const auto& reference = get<0>(pair)->get<ReferenceAccess>();

		if (!reference.isDummy())
		{
			auto destination = symbolTable.lookup(reference.getName());
			auto& value = get<1>(pair);
			builder.create<StoreOp>(builder.getUnknownLoc(), value, destination);
		}
	}
}

void MlirLowerer::lower(const modelica::IfStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);

	auto insertionPoint = builder.saveInsertionPoint();
	size_t blocks = statement.size();

	for (size_t i = 0; i < blocks; i++)
	{
		const auto& conditionalBlock = statement[i];
		auto condition = lower<modelica::Expression>(conditionalBlock.getCondition())[0];
		bool elseBlock = i < blocks - 1;
		auto ifOp = builder.create<scf::IfOp>(builder.getUnknownLoc(), condition, elseBlock);
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (const auto& stmnt : conditionalBlock)
			lower(stmnt);

		if (elseBlock)
			builder.setInsertionPointToStart(&ifOp.elseRegion().front());
	}

	builder.restoreInsertionPoint(insertionPoint);
}

void MlirLowerer::lower(const modelica::ForStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
}

void MlirLowerer::lower(const modelica::WhileStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
}

void MlirLowerer::lower(const modelica::WhenStatement& statement)
{
	ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
}

void MlirLowerer::lower(const modelica::BreakStatement& statement)
{

}

void MlirLowerer::lower(const modelica::ReturnStatement& statement)
{

}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<Expression>(const modelica::Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return lower<deconst>(expression);
	});
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Operation>(const modelica::Expression& expression)
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
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Constant>(const modelica::Expression& expression)
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
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::ReferenceAccess>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::ReferenceAccess>());
	const auto& reference = expression.get<modelica::ReferenceAccess>();
	auto var = symbolTable.lookup(reference.getName());
	return { builder.create<LoadOp>(builder.getUnknownLoc(), var) };
}

template<>
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Call>(const modelica::Expression& expression)
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
MlirLowerer::Container<mlir::Value> MlirLowerer::lower<modelica::Tuple>(const modelica::Expression& expression)
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
