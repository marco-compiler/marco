#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <modelica/mlirlowerer/LowerToLLVM.hpp>
#include <modelica/mlirlowerer/LowerToStandard.hpp>
#include <modelica/mlirlowerer/MlirLowerer.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>

using namespace llvm;
using namespace mlir;
using namespace modelica;
using namespace std;

mlir::LogicalResult modelica::convertToLLVMDialect(mlir::MLIRContext* context, mlir::ModuleOp module, ModelicaOptions options)
{
	mlir::PassManager passManager(context);

	// Convert the Modelica dialect to the Standard one
	passManager.addPass(createModelicaToStandardLoweringPass());

	// Convert the output buffers to input buffers, in order to delegate the
	// buffer allocation to the caller.
	passManager.addPass(createBufferResultsToOutParamsPass());

	// Convert vector operations to loops
	passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

	// Lower the SCF operations
	passManager.addPass(createLowerToCFGPass());

	//passManager.addNestedPass<FuncOp>(createMemRefDataFlowOptPass());

	// Conversion to LLVM dialect
	ModelicaToLLVMLoweringOptions modelicaToLLVMOptions;
	modelicaToLLVMOptions.useBarePtrCallConv = options.useBarePtrCallConv;
	passManager.addPass(createModelicaToLLVMLoweringPass(modelicaToLLVMOptions));

	return passManager.run(module);
}

Reference::Reference() : builder(nullptr), value(nullptr), reader(nullptr)
{
}

Reference::Reference(mlir::OpBuilder* builder,
										 mlir::Value value,
										 std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader)
		: builder(builder),
			value(std::move(value)),
			reader(std::move(reader))
{
}

mlir::Value Reference::operator*()
{
	return reader(builder, value);
}

mlir::Value Reference::getReference() const
{
	return value;
}

Reference Reference::ssa(mlir::OpBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value
			{
				return value;
			});
}

Reference Reference::memref(mlir::OpBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value
			{
				mlir::Type type = value.getType();
				assert(type.isa<MemRefType>());
				auto memRefType = type.cast<MemRefType>();

				if (memRefType.getNumElements() == 1)
					return builder->create<mlir::LoadOp>(value.getLoc(), value);

				mlir::VectorType vectorType = mlir::VectorType::get(memRefType.getShape(), memRefType.getElementType());
				mlir::Value zeroValue = builder->create<ConstantOp>(value.getLoc(), builder->getIndexAttr(0));
				SmallVector<mlir::Value, 3> indexes(memRefType.getRank(), zeroValue);
				return builder->create<AffineVectorLoadOp>(value.getLoc(), vectorType, value, indexes);
			});
}

MlirLowerer::MlirLowerer(mlir::MLIRContext& context, ModelicaOptions options)
		: builder(&context), options(move(options))
{
	// Check that the required dialects have been previously registered
	context.loadDialect<ModelicaDialect>();
	context.loadDialect<StandardOpsDialect>();
	context.loadDialect<scf::SCFDialect>();
	context.loadDialect<linalg::LinalgDialect>();
	context.loadDialect<AffineDialect>();
	context.loadDialect<mlir::vector::VectorDialect>();
	context.loadDialect<LLVM::LLVMDialect>();

	if (options.x64)
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
	mlir::Type destinationBase = destination;

	if (source.isa<ShapedType>())
	{
		sourceBase = source.cast<ShapedType>().getElementType();
		auto sourceShape = source.cast<ShapedType>().getShape();

		if (destination.isa<ShapedType>())
		{
			auto destinationShape = destination.cast<ShapedType>().getShape();
			destinationBase = destination.cast<ShapedType>().getElementType();
			assert(all_of(llvm::zip(sourceShape, destinationShape),
										[](const auto& pair)
										{
											return get<0>(pair) == get<1>(pair);
										}));

			destination = mlir::VectorType::get(destinationShape, destinationBase);
		}
		else
		{
			destination = mlir::VectorType::get(sourceShape, destinationBase);
		}
	}

	if (sourceBase == destinationBase)
		return value;

	if (sourceBase.isSignlessInteger())
	{
		if (destinationBase.isF32() || destinationBase.isF64())
			return builder.create<SIToFPOp>(value.getLoc(), value, destination);

		if (destinationBase.isIndex())
			return builder.create<IndexCastOp>(value.getLoc(), value, destination);
	}
	else if (sourceBase.isF32() || sourceBase.isF64())
	{
		if (destinationBase.isSignlessInteger())
			return builder.create<FPToSIOp>(value.getLoc(), value, destination);
	}
	else if (sourceBase.isIndex())
	{
		if (destinationBase.isSignlessInteger())
			return builder.create<IndexCastOp>(value.getLoc(), value, integerType);

		if (destinationBase.isF32() || destinationBase.isF64())
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

	for (const auto& member : foo.getArgs())
	{
		argNames.emplace_back(member->getName());
		argTypes.emplace_back(lower(member->getType()));
	}

	SmallVector<llvm::StringRef, 3> returnNames;
	SmallVector<mlir::Type, 3> returnTypes;
	auto outputMembers = foo.getResults();

	for (const auto& member : outputMembers)
	{
		returnNames.emplace_back(member->getName());
		returnTypes.emplace_back(lower(member->getType()));
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
		const auto type = value.getType();

		if (type.isa<mlir::MemRefType>())
			symbolTable.insert(name, Reference::memref(&builder, value));
		else
			symbolTable.insert(name, Reference::ssa(&builder, value));
	}

	// Set the insertion point in the builder to the beginning of the function
	// body, it will be used throughout the codegen to create operations in this
	// function.
	builder.setInsertionPointToStart(&entryBlock);

	// Initialize members
	for (const auto& member : foo.getMembers())
		lower(member);

	// Emit the body of the function
	const auto& algorithm = foo.getAlgorithms()[0];

	// Create the variable to be checked for an early return
	auto algorithmLocation = loc(algorithm.getLocation());
	mlir::Value returnCondition = builder.create<AllocaOp>(algorithmLocation, MemRefType::get({}, builder.getI1Type()));
	symbolTable.insert(algorithm.getReturnCheckName(), Reference::memref(&builder, returnCondition));
	mlir::Value falseValue = builder.create<ConstantOp>(algorithmLocation, builder.getBoolAttr(false));
	builder.create<StoreOp>(algorithmLocation, falseValue, returnCondition);

	// Lower the statements
	lower(foo.getAlgorithms()[0]);

	// Return statement
	std::vector<mlir::Value> results;

	for (const auto& member : outputMembers)
	{
		auto ptr = symbolTable.lookup(member->getName());

		if (member->getType().isScalar())
			results.push_back(*ptr);
		else
			results.push_back(ptr.getReference());
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

	if (member.isInput())
		return;

	auto type = lower(member.getType());
	mlir::Value var;

	if (type.isa<MemRefType>())
	{
		auto shape = type.cast<MemRefType>().getShape();
		auto baseType = type.cast<MemRefType>().getElementType();
		var = builder.create<AllocaOp>(location, MemRefType::get(shape, baseType));
	}
	else
	{
		var = builder.create<AllocaOp>(location, MemRefType::get({}, type));
	}

	symbolTable.insert(member.getName(), Reference::memref(&builder, var));

	if (member.hasInitializer())
	{
		auto reference = lower<modelica::Expression>(member.getInitializer())[0];
		builder.create<AssignmentOp>(loc(member.getInitializer().getLocation()), reference.getReference(), var);
	}
}

void MlirLowerer::lower(const modelica::Algorithm& algorithm)
{
	for (const auto& statement : algorithm)
		lower(statement);
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
		auto destination = lower<modelica::Expression>(get<0>(pair))[0];
		auto value = get<1>(pair);

		bool valueIsArray = false;
		mlir::Type valueType = value.getReference().getType();

		builder.create<AssignmentOp>(location, value.getReference(), destination.getReference());
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

		auto ifOp = builder.create<scf::IfOp>(loc(statement.getLocation()), *condition, elseBlock);

		if (firstOp == nullptr)
			firstOp = ifOp;

		// "Then" block
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (const auto& stmnt : conditionalBlock)
			lower(stmnt);

		if (i > 0)
		{
			builder.setInsertionPointAfter(ifOp);
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

	// Variable to be set when calling "break"
	mlir::Value breakCondition = builder.create<AllocaOp>(location, MemRefType::get({}, builder.getI1Type()));
	symbolTable.insert(statement.getBreakCheckName(), Reference::memref(&builder, breakCondition));
	mlir::Value falseValue = builder.create<ConstantOp>(location, builder.getBoolAttr(false));
	builder.create<StoreOp>(location, falseValue, breakCondition);

	// Variable to be set when calling "return"
	mlir::Value returnCondition = symbolTable.lookup(statement.getReturnCheckName()).getReference();

	const auto& induction = statement.getInduction();
	auto lowerBound = cast(*lower<modelica::Expression>(induction.getBegin())[0], builder.getIndexType());
	auto forOp = builder.create<ForOp>(location, breakCondition, returnCondition, lowerBound);

	{
		// Check the loop condition
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.condition().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.condition().front());

		auto upperBound = cast(*lower<modelica::Expression>(induction.getEnd())[0], builder.getIndexType());
		mlir::Value condition = builder.create<CmpIOp>(location, CmpIPredicate::slt, forOp.condition().front().getArgument(0), upperBound);
		builder.create<ConditionOp>(location, condition, *symbolTable.lookup(induction.getName()));
	}

	{
		// Body
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.body().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.body().front());

		for (const auto& stmnt : statement)
			lower(stmnt);

		builder.create<YieldOp>(location, *symbolTable.lookup(induction.getName()));
	}

	{
		// Step
		ScopedHashTableScope<StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.step().front().getArgument(0)));

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

	// Variable to be set when calling "break"
	mlir::Value breakCondition = builder.create<AllocaOp>(location, MemRefType::get({}, builder.getI1Type()));
	symbolTable.insert(statement.getBreakCheckName(), Reference::memref(&builder, breakCondition));
	mlir::Value falseValue = builder.create<ConstantOp>(location, builder.getBoolAttr(false));
	builder.create<StoreOp>(location, falseValue, breakCondition);

	// Variable to be set when calling "return"
	mlir::Value returnCondition = symbolTable.lookup(statement.getReturnCheckName()).getReference();

	// Create the operation
	auto whileOp = builder.create<WhileOp>(location, breakCondition, returnCondition);

	{
		// Condition
		ScopedHashTableScope<StringRef, Reference> varScope(symbolTable);
		mlir::Block* conditionBlock = &whileOp.condition().front();
		builder.setInsertionPointToStart(conditionBlock);
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

		builder.create<YieldOp>(location);
	}

	// Keep populating after the while operation
	builder.setInsertionPointAfter(whileOp);
}

void MlirLowerer::lower(const modelica::WhenStatement& statement)
{

}

void MlirLowerer::lower(const modelica::BreakStatement& statement)
{
	auto location = loc(statement.getLocation());
	mlir::Value trueValue = builder.create<ConstantOp>(location, builder.getBoolAttr(true));
	mlir::Value breakCondition = symbolTable.lookup(statement.getBreakCheckName()).getReference();
	builder.create<StoreOp>(location, trueValue, breakCondition);
}

void MlirLowerer::lower(const modelica::ReturnStatement& statement)
{
	auto location = loc(statement.getLocation());
	mlir::Value trueValue = builder.create<ConstantOp>(location, builder.getBoolAttr(true));
	mlir::Value returnCondition = symbolTable.lookup(statement.getReturnCheckName()).getReference();
	builder.create<StoreOp>(location, trueValue, returnCondition);
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
		auto arg = lower<modelica::Expression>(operation[0])[0].getReference();
		mlir::Value result = builder.create<modelica::NegateOp>(location, arg);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::add)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::AddOp>(location, args);
		result = cast(result, resultType);

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subtract)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::SubOp>(location, args);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::multiply)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::MulOp>(location, args);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::divide)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::DivOp>(location, args);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::powerOf)
	{
		mlir::Value base = cast(*lower<modelica::Expression>(operation[0])[0], floatType);
		mlir::Value exponent = cast(*lower<modelica::Expression>(operation[1])[0], floatType);
		mlir::Value result = builder.create<PowFOp>(location, base, exponent);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::equal)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<EqOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::different)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<NotEqOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greater)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<GtOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greaterEqual)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<GteOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::less)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<LtOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::lessEqual)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<LteOp>(location, args[0], args[1]);
		result = cast(result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::ifelse)
	{
		mlir::Value condition = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value trueValue = cast(*lower<modelica::Expression>(operation[1])[0], resultType);
		mlir::Value falseValue = cast(*lower<modelica::Expression>(operation[2])[0], resultType);
		mlir::Value result = builder.create<SelectOp>(location, condition, trueValue, falseValue);
		return { Reference::ssa(&builder, result) };
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
		return { Reference::ssa(&builder, result) };
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
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subscription)
	{
		// Base pointer
		auto memref = lower<modelica::Expression>(operation[0])[0];
		mlir::Type type = memref.getReference().getType();
		assert(type.isa<MemRefType>());
		auto memrefType = type.cast<MemRefType>();

		// Create the subview
		mlir::Value zeroValue = builder.create<ConstantOp>(location, builder.getIndexAttr(0));
		mlir::Value oneValue = builder.create<ConstantOp>(location, builder.getIndexAttr(1));

		SmallVector<long, 3> resultIndexes;

		SmallVector<long, 3> staticOffsets;
		SmallVector<mlir::Value, 3> dynamicOffsets;
		SmallVector<long, 3> staticSizes;
		SmallVector<long, 3> staticStrides;

		SmallVector<mlir::Value, 3> indexes;

		for (size_t i = 1; i < operation.argumentsCount(); i++)
		{
			auto subscript = *lower<modelica::Expression>(operation[i])[0];
			mlir::Value index = cast(subscript, builder.getIndexType());

			staticOffsets.push_back(ShapedType::kDynamicStrideOrOffset);
			dynamicOffsets.push_back(index);

			// Set the dimension to 1 in order to get a slice along that dimension
			staticSizes.push_back(1);

			// The elements are supposed to be spaced by 1 element size in each dimension
			staticStrides.push_back(1);
		}

		for (long i = staticSizes.size(); i < memrefType.getRank(); i++)
		{
			// Start at the beginning of that dimension
			staticOffsets.push_back(0);

			// The remaining sizes will be as big as the original ones
			long size = memrefType.getDimSize(i);
			staticSizes.push_back(size);
			resultIndexes.push_back(size);

			// Again, strides supposed to be 1
			staticStrides.push_back(1);
		}

		SmallVector<long, 3> mapStrides;

		for (size_t i = 0; i < resultIndexes.size(); i++)
			mapStrides.push_back(1);

		auto map = makeStridedLinearLayoutMap(
				mapStrides,
				dynamicOffsets.empty() ? 0 : ShapedType::kDynamicStrideOrOffset,
				builder.getContext());

		mlir::Value sourceView = builder.create<SubViewOp>(
				location,
				MemRefType::get(resultIndexes, memrefType.getElementType(), map),
				memref.getReference(),
				staticOffsets, staticSizes, staticStrides,
				dynamicOffsets, ValueRange(), ValueRange());

		return { Reference::memref(&builder, sourceView) };
	}

	if (kind == OperationKind::memberLookup)
	{
		// TODO
		return { Reference::ssa(&builder, nullptr) };
	}

	assert(false && "Unexpected operation");
	return { Reference::ssa(&builder, nullptr) };
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

	return { Reference::ssa(&builder, value) };
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
		results.emplace_back(Reference::ssa(&builder, result));

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

	bool containsInteger = false;
	bool containsFloat = false;

	for (const auto& arg : operation)
	{
		const auto& type = arg.getType();

		// For now, we only support operation between built-in types.
		// In future, this should be extended to support struct types.
		assert(type.isA<modelica::BuiltInType>());

		if (type.get<BuiltInType>() == BuiltInType::Integer)
			containsInteger = true;
		else if (type.get<BuiltInType>() == BuiltInType::Float)
			containsFloat = true;

		args.push_back(*lower<modelica::Expression>(arg)[0]);
	}

	// Convert the arguments to a common type.
	// If any of the arguments is a float, also the others must be floats, in
	// order to preserve correctness. If a value is a boolean, it is first
	// extended to an integer in first place, and then, if needed, to a float.

	for (const auto& arg : args)
	{
		if (containsFloat)
			castedArgs.push_back(cast(arg, floatType));
		else if (containsInteger)
			castedArgs.push_back(cast(arg, integerType));
		else
			castedArgs.push_back(arg);
	}

	return castedArgs;
}
