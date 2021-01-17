#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ModelicaToStandard.hpp>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

mlir::LogicalResult modelica::convertToLLVMDialect(mlir::MLIRContext* context, mlir::ModuleOp module)
{
	mlir::PassManager passManager(context);
	passManager.addPass(createModelicaToStdPass());
	passManager.addPass(createBufferResultsToOutParamsPass());
	passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
	passManager.addPass(createLowerToCFGPass());

	LowerToLLVMOptions llvmLoweringOptions;
	llvmLoweringOptions.emitCWrappers = true;
	passManager.addPass(createLowerToLLVMPass(llvmLoweringOptions));

	return passManager.run(module);
}

Reference::Reference() : builder(nullptr), value(nullptr), isPtr(false)
{
}

mlir::Value Reference::operator*()
{
	if (isPtr)
		return builder->create<LoadOp>(builder->getUnknownLoc(), value, indexes);

	return value;
}

mlir::Value Reference::getValue() const
{
	return value;
}

mlir::ValueRange Reference::getIndexes() const
{
	return indexes;
}

MlirLowerer::MlirLowerer(mlir::MLIRContext& context, bool x64)
		: builder(&context)
{
	// Check that the required dialects have been previously registered
	context.loadDialect<ModelicaDialect>();
	context.loadDialect<StandardOpsDialect>();
	context.loadDialect<LLVM::LLVMDialect>();

	if (x64)
	{
		integerType = builder.getI64Type();
		floatType = builder.getF64Type();
	}
	else
	{
		integerType = builder.getI32Type();
		floatType = builder.getF32Type();
	}
}

mlir::OpBuilder& MlirLowerer::getOpBuilder()
{
	return builder;
}

mlir::Location MlirLowerer::loc(SourcePosition location)
{
	return builder.getFileLineColLoc(
			builder.getIdentifier(*location.file),
			location.line,
			location.column);
}

mlir::Value MlirLowerer::cast(mlir::Value value, mlir::Type destination)
{
	auto source = value.getType();

	if (source == destination)
		return value;

	mlir::Type sourceBase = source;

	if (sourceBase.isSignlessInteger())
	{
		if (destination.isF32() || destination.isF64())
			return builder.create<SIToFPOp>(value.getLoc(), value, destination);

		if (destination.isIndex())
			return builder.create<IndexCastOp>(value.getLoc(), value, destination);
	}
	else if (source.isF32() || source.isF64())
	{
		if (destination.isSignlessInteger())
			return builder.create<FPToSIOp>(value.getLoc(), value, destination);
	}
	else if (source.isIndex())
	{
		if (destination.isSignlessInteger())
			return builder.create<IndexCastOp>(value.getLoc(), value, integerType);

		if (destination.isF32() || destination.isF64())
			return cast(builder.create<IndexCastOp>(value.getLoc(), value, integerType), destination);
	}

	assert(false && "Unsupported type conversion");
	return nullptr;
}

mlir::ModuleOp MlirLowerer::lower(llvm::ArrayRef<const modelica::ClassContainer> classes)
{
	mlir::ModuleOp module = ModuleOp::create(builder.getUnknownLoc());

	for (const auto& cls : classes)
	{
		auto* op = cls.visit([&](auto& obj) -> mlir::Operation* { return lower(obj); });

		if (op != nullptr)
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
	ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);

	auto location = loc(foo.getLocation());

	SmallVector<llvm::StringRef, 3> argNames;
	SmallVector<mlir::Type, 3> argTypes;
	SmallVector<llvm::StringRef, 3> returnNames;
	SmallVector<mlir::Type, 3> returnTypes;

	auto outputMembers = foo.getResults();

	for (const auto& member : outputMembers)
	{
		returnNames.emplace_back(member->getName());
		returnTypes.emplace_back(lower(member->getType()));
	}

	for (const auto& member : foo.getArgs())
	{
		argNames.emplace_back(member->getName());
		argTypes.emplace_back(lower(member->getType()));
	}

	auto functionType = builder.getFunctionType(argTypes, returnTypes);
	auto function = mlir::FuncOp::create(location, foo.getName(), functionType);

	// Start the body of the function.
	// In MLIR the entry block of the function is special: it must have the same
	// argument list as the function itself.
	auto &entryBlock = *function.addEntryBlock();

	// Declare all the function arguments in the symbol table
	for (const auto& pair : llvm::zip(argNames, entryBlock.getArguments())) {
		const auto& name = get<0>(pair);
		const auto& value = get<1>(pair);
		symbolTable.insert(name, Reference(builder, value, false));
	}

	// Create a separate block for the return of the results.
	// This way, a return operation in the middle of the function body can
	// jump directly to that last block and return the values to the callee.
	Block* returnBlock = function.addBlock();

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Initialize members
	for (const auto& member : foo.getMembers())
		lower(member);

	// Emit the body of the function
	lower(foo.getAlgorithms()[0]);

	builder.create<BranchOp>(builder.getUnknownLoc(), returnBlock);

	// Return statement
	builder.setInsertionPointToStart(returnBlock);
	std::vector<mlir::Value> results;

	for (const auto& member : outputMembers)
	{
		auto ptr = symbolTable.lookup(member->getName());

		if (member->getType().isScalar())
			results.push_back(*ptr);
		else
			results.push_back(ptr.getValue());
	}

	builder.create<ReturnOp>(location, results);
	return function;
}

mlir::Type MlirLowerer::lower(const modelica::Type& type)
{
	auto visitor = [&](auto& obj) -> mlir::Type
	{
		auto baseType = lower(obj);

		if (!type.isScalar())
		{
			const auto& dimensions = type.getDimensions();
			SmallVector<long, 3> shape(dimensions.begin(), dimensions.end());
			return mlir::MemRefType::get(shape, baseType);
		}

		return baseType;
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
			return integerType;
		case BuiltInType::Float:
			return floatType;
		case BuiltInType::Boolean:
			return builder.getI1Type();
		default:
			assert(false && "Unexpected type");
			return builder.getNoneType();
	}
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
	auto location = loc(member.getLocation());

	if (!member.getType().isScalar() && member.isInput())
		return;

	auto type = lower(member.getType());
	mlir::Value var;

	if (type.isa<ShapedType>())
	{
		auto shape = type.cast<ShapedType>().getShape();
		auto baseType = type.cast<ShapedType>().getElementType();
		var = builder.create<AllocaOp>(location, MemRefType::get(shape, baseType));
		symbolTable.insert(member.getName(), Reference(builder, var, true));
	}
	else
	{
		var = builder.create<AllocaOp>(location, MemRefType::get({}, type));

		// Input variables already have an associated value. Copy it to the stack.
		if (member.isInput())
		{
			auto value = *symbolTable.lookup(member.getName());
			builder.create<StoreOp>(value.getLoc(), value, var);
		}

		symbolTable.insert(member.getName(), Reference(builder, var, true));

		if (member.hasInitializer())
		{
			auto reference = lower<modelica::Expression>(member.getInitializer())[0];
			builder.create<StoreOp>(loc(member.getInitializer().getLocation()), *reference, var);
		}
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
	auto location = loc(statement.getLocation());
	auto destinations = statement.getDestinations();
	auto values = lower<modelica::Expression>(statement.getExpression());
	assert(values.size() == destinations.size() && "Unequal number of destinations and results");

	for (auto pair : zip(destinations, values))
	{
		auto& destination = lower<modelica::Expression>(get<0>(pair))[0];
		auto value = *get<1>(pair);

		if (value.getType().isa<ShapedType>())
			builder.create<MemCopyOp>(location, value, destination.getValue());
		else
			builder.create<StoreOp>(location, value, destination.getValue(), destination.getIndexes());
	}
}

void MlirLowerer::lower(const modelica::IfStatement& statement)
{
	// Each conditional blocks creates an If operation, but we need to keep
	// track of the first one in order to restore the insertion point right
	// after that when we have finished to lower all the blocks.
	mlir::Operation* firstOp = nullptr;

	size_t blocks = statement.size();

	for (size_t i = 0; i < blocks; i++)
	{
		ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);
		const auto& conditionalBlock = statement[i];
		auto condition = lower<modelica::Expression>(conditionalBlock.getCondition())[0];

		// The last conditional block can be at most an originally equivalent
		// "else" block, and thus doesn't need a lowered else block.
		bool elseBlock = i < blocks - 1;

		auto ifOp = builder.create<IfOp>(loc(statement.getLocation()), *condition, elseBlock);

		if (firstOp == nullptr)
			firstOp = ifOp;

		// "Then" block
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (const auto& stmnt : conditionalBlock)
			lower(stmnt);

		auto& lastOp = ifOp.thenRegion().back().getOperations().back();

		if (lastOp.isKnownNonTerminator())
			builder.create<YieldOp>(builder.getUnknownLoc());

		if (i > 0)
		{
			builder.setInsertionPointAfter(ifOp);
			builder.create<YieldOp>(builder.getUnknownLoc());
		}

		// The next conditional blocks will be placed as new If operations
		// nested inside the "else" block.
		if (elseBlock)
			builder.setInsertionPointToStart(&ifOp.elseRegion().front());
	}

	builder.setInsertionPointAfter(firstOp);
}

void MlirLowerer::lower(const modelica::ForStatement& statement)
{
	ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);

	auto location = loc(statement.getLocation());

	const auto& induction = statement.getInduction();
	auto lowerBound = cast(*lower<modelica::Expression>(induction.getBegin())[0], builder.getIndexType());
	auto forOp = builder.create<ForOp>(location, lowerBound);

	{
		// Check the loop condition
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference(builder, forOp.condition().front().getArgument(0), false));

		builder.setInsertionPointToStart(&forOp.condition().front());

		auto upperBound = cast(*lower<modelica::Expression>(induction.getEnd())[0], builder.getIndexType());
		auto condition = builder.create<CmpIOp>(location, CmpIPredicate::slt, forOp.condition().front().getArgument(0), upperBound);
		builder.create<ConditionOp>(location, condition, *symbolTable.lookup(induction.getName()));
	}

	{
		// Body
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference(builder, forOp.body().front().getArgument(0), false));

		builder.setInsertionPointToStart(&forOp.body().front());

		for (const auto& stmnt : statement)
			lower(stmnt);

		auto& lastBodyOp = forOp.body().back().getOperations().back();

		if (lastBodyOp.isKnownNonTerminator())
			builder.create<YieldOp>(location, *symbolTable.lookup(induction.getName()));
	}

	{
		// Step
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference(builder, forOp.step().front().getArgument(0), false));

		builder.setInsertionPointToStart(&forOp.step().front());

		mlir::Value step = builder.create<ConstantOp>(location, builder.getIndexAttr(1));
		mlir::Value incremented = builder.create<AddIOp>(location, forOp.step().front().getArgument(0), step);
		builder.create<YieldOp>(location, incremented);
	}

	builder.setInsertionPointAfter(forOp);
}

void MlirLowerer::lower(const modelica::WhileStatement& statement)
{
	auto location = loc(statement.getLocation());
	auto whileOp = builder.create<WhileOp>(location);

	{
		// Condition
		ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);
		builder.setInsertionPointToStart(&whileOp.condition().front());
		const auto& condition = statement.getCondition();

		builder.create<ConditionOp>(
				loc(condition.getLocation()),
				*lower<modelica::Expression>(condition)[0]);
	}

	{
		// Body
		ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);
		builder.setInsertionPointToStart(&whileOp.body().front());

		for (const auto& stmnt : statement)
			lower(stmnt);

		auto& lastBodyOp = whileOp.body().back().getOperations().back();

		if (lastBodyOp.isKnownNonTerminator())
			builder.create<YieldOp>(builder.getUnknownLoc());
	}

	// Keep populating after the while operation
	builder.setInsertionPointAfter(whileOp);
}

void MlirLowerer::lower(const modelica::WhenStatement& statement)
{
	// TODO
}

void MlirLowerer::lower(const modelica::BreakStatement& statement)
{
	mlir::Operation* loop = builder.getInsertionBlock()->getParentOp();

	while (!loop->hasTrait<BreakableLoop::Trait>())
		loop = loop->getParentOp();

	auto location = loc(statement.getLocation());
	auto op = dyn_cast<BreakableLoop>(loop);
	builder.create<BreakOp>(location, &op.exit().front());
}

void MlirLowerer::lower(const modelica::ReturnStatement& statement)
{
	auto location = loc(statement.getLocation());
	mlir::Operation* op = builder.getInsertionBlock()->getParentOp();

	while (!op->hasTrait<mlir::OpTrait::FunctionLike>())
		op = op->getParentOp();

	// Jump to the "return" block of the function
	auto* continuation = &op->getRegion(0).back();

	builder.create<BranchOp>(location, continuation);
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<Expression>(const modelica::Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return lower<deconst>(expression);
	});
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Operation>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::Operation>());
	const auto& operation = expression.get<modelica::Operation>();
	auto kind = operation.getKind();
	mlir::Type resultType = lower(expression.getType());
	mlir::Location location = loc(expression.getLocation());

	if (kind == OperationKind::negate)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 1);
		mlir::Value result = builder.create<modelica::NegateOp>(location, args[0]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::add)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::AddOp>(location, args);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::subtract)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::SubOp>(location, args);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::multiply)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::MulOp>(location, args);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::divide)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::DivOp>(location, args);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::powerOf)
	{
		mlir::Value base = cast(*lower<modelica::Expression>(operation[0])[0], floatType);
		mlir::Value exponent = cast(*lower<modelica::Expression>(operation[1])[0], floatType);
		mlir::Value result = builder.create<PowFOp>(location, base, exponent);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::equal)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<EqOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::different)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<NotEqOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::greater)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<GtOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::greaterEqual)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<GteOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::less)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<LtOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::lessEqual)
	{
		auto args = lowerOperationArgs(operation);
		assert(args.size() == 2);
		mlir::Value result = builder.create<LteOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::ifelse)
	{
		mlir::Value condition = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value trueValue = cast(*lower<modelica::Expression>(operation[1])[0], resultType);
		mlir::Value falseValue = cast(*lower<modelica::Expression>(operation[2])[0], resultType);
		mlir::Value result = builder.create<SelectOp>(location, condition, trueValue, falseValue);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::land)
	{
		mlir::Value lhs = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value rhs = *lower<modelica::Expression>(operation[1])[0];

		unsigned int lhsBitWidth = lhs.getType().getIntOrFloatBitWidth();
		unsigned int rhsBitWidth = lhs.getType().getIntOrFloatBitWidth();

		if (lhsBitWidth < rhsBitWidth)
			lhs = builder.create<SignExtendIOp>(location, lhs, rhs.getType());
		else if (lhsBitWidth > rhsBitWidth)
			rhs = builder.create<SignExtendIOp>(location, rhs, lhs.getType());

		mlir::Value result = builder.create<AndOp>(location, lhs, rhs);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::lor)
	{
		mlir::Value lhs = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value rhs = *lower<modelica::Expression>(operation[1])[0];

		unsigned int lhsBitWidth = lhs.getType().getIntOrFloatBitWidth();
		unsigned int rhsBitWidth = lhs.getType().getIntOrFloatBitWidth();

		if (lhsBitWidth < rhsBitWidth)
			lhs = builder.create<SignExtendIOp>(location, lhs, rhs.getType());
		else if (lhsBitWidth > rhsBitWidth)
			rhs = builder.create<SignExtendIOp>(location, rhs, lhs.getType());

		mlir::Value result = builder.create<OrOp>(location, lhs, rhs);
		return { Reference(builder, result, false) };
	}

	if (kind == OperationKind::subscription)
	{
		auto var = lower<modelica::Expression>(operation[0])[0];
		SmallVector<mlir::Value, 3> indexes;

		for (size_t i = 1; i < operation.argumentsCount(); i++)
		{
			auto subscript = *lower<modelica::Expression>(operation[i])[0];
			auto index = cast(subscript, builder.getIndexType());
			indexes.push_back(index);
		}

		return { Reference(builder, var.getValue(), true, indexes) };
	}

	if (kind == OperationKind::memberLookup)
	{
		// TODO
		return { Reference(builder, nullptr, false) };
	}

	assert(false && "Unexpected operation");
	return { Reference(builder, nullptr, false) };
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Constant>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::Constant>());
	const auto& constant = expression.get<modelica::Constant>();

	auto value = builder.create<ConstantOp>(
			loc(expression.getLocation()),
			constantToType(constant),
			constant.visit([&](const auto& obj) { return getAttribute(obj); }));

	return { Reference(builder, value, false) };
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::ReferenceAccess>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::ReferenceAccess>());
	const auto& reference = expression.get<modelica::ReferenceAccess>();
	return { symbolTable.lookup(reference.getName()) };
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Call>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::Call>());
	const auto& call = expression.get<modelica::Call>();
	const auto& function = call.getFunction();

	SmallVector<mlir::Value, 3> args;

	for (const auto& arg : call)
	{
		auto& reference = lower<modelica::Expression>(arg)[0];
		args.push_back(*reference);
	}

	auto op = builder.create<CallOp>(
			loc(expression.getLocation()),
			function.get<ReferenceAccess>().getName(),
			lower(function.getType()),
			args);

	Container<Reference> results;

	for (auto result : op.getResults())
		results.emplace_back(builder, result, false);

	return results;
}

template<>
MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Tuple>(const modelica::Expression& expression)
{
	assert(expression.isA<modelica::Tuple>());
	const auto& tuple = expression.get<modelica::Tuple>();
	Container<Reference> result;

	for (auto& exp : tuple)
	{
		auto values = lower<modelica::Expression>(expression);

		// The only way to have multiple returns is to call a function, but this
		// is forbidden in a tuple declaration. In fact, a tuple is just a
		// container of references.
		assert(values.size() == 1);
		result.emplace_back(values[0]);
	}

	return result;
}

MlirLowerer::Container<mlir::Value> MlirLowerer::lowerOperationArgs(const modelica::Operation& operation)
{
	SmallVector<mlir::Value, 3> args;
	SmallVector<mlir::Value, 3> castedArgs;
	bool containsFloat = false;

	for (const auto& arg : operation)
	{
		// For now, we only support operation between built-in types.
		// In future, this should be extended to support struct types.
		assert(arg.getType().isA<modelica::BuiltInType>());

		if (arg.getType().get<BuiltInType>() == BuiltInType::Float)
			containsFloat = true;

		args.push_back(*lower<modelica::Expression>(arg)[0]);
	}

	// Convert the arguments to a common type.
	// If any of the arguments is a float, also the others must be floats, in
	// order to preserve correctness. If a value is a boolean, it is first
	// extended to an integer in first place, and then, if needed, to a float.

	for (const auto& arg : args)
	{
		mlir::Value castedArg = arg;

		// Bool --> Int
		if (castedArg.getType().isInteger(1))
			castedArg = builder.create<SignExtendIOp>(arg.getLoc(), castedArg, integerType);

		// Index --> Integer
		if (castedArg.getType().isIndex())
			castedArg = builder.create<IndexCastOp>(arg.getLoc(), castedArg, integerType);

		// Integer --> Float
		if (containsFloat && castedArg.getType().isSignlessInteger())
			castedArg = builder.create<SIToFPOp>(arg.getLoc(), castedArg, floatType);

		castedArgs.push_back(castedArg);
	}

	return castedArgs;
}
