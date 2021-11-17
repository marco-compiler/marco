#include <llvm/ADT/SmallVector.h>
#include <marco/frontend/AST.h>
#include <marco/mlirlowerer/CodeGen.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>
#include <mlir/IR/Verifier.h>

using namespace marco;
using namespace frontend;
using namespace marco::codegen;
using namespace modelica;
using namespace std;

Reference::Reference()
		: builder(nullptr),
			value(nullptr),
			reader(nullptr)
{
}

Reference::Reference(mlir::OpBuilder* builder,
										 mlir::Value value,
										 std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader,
										 std::function<void(mlir::OpBuilder*, Reference&, mlir::Value)> writer)
		: builder(builder),
			value(std::move(value)),
			reader(std::move(reader)),
			writer(std::move(writer))
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

void Reference::set(mlir::Value v)
{
	writer(builder, *this, v);
}

Reference Reference::ssa(mlir::OpBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value {
				return value;
			},
			[](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
				assert(false && "Can't assign value to SSA operand");
			});
}

Reference Reference::memory(mlir::OpBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value {
				auto arrayType = value.getType().cast<ArrayType>();

				// We can load the value only if it's a pointer to a scalar.
				// Otherwise, return the array.

				if (arrayType.getShape().empty())
					return builder->create<LoadOp>(value.getLoc(), value);

				return value;
			},
			[&](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
				assert(destination.value.getType().isa<ArrayType>());
				builder->create<AssignmentOp>(value.getLoc(), value, destination.getReference());
			});
}

Reference Reference::member(mlir::OpBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value
			{
				auto memberType = value.getType().cast<MemberType>();

				if (memberType.getShape().empty())
					return builder->create<MemberLoadOp>(value.getLoc(), memberType.getElementType(), value);

				return builder->create<MemberLoadOp>(value.getLoc(), memberType.toArrayType(), value);
			},
			[](mlir::OpBuilder* builder, Reference& destination, mlir::Value value) {
				builder->create<MemberStoreOp>(value.getLoc(), destination.value, value);
			});
}

MLIRLowerer::MLIRLowerer(mlir::MLIRContext& context, ModelicaOptions options)
		: builder(&context),
			options(options)
{
	context.loadDialect<ModelicaDialect>();
	context.loadDialect<mlir::StandardOpsDialect>();
}

mlir::Location MLIRLowerer::loc(SourcePosition location)
{
	return mlir::FileLineColLoc::get(
			builder.getIdentifier(*location.file),
			location.line,
			location.column);
}

mlir::Location MLIRLowerer::loc(SourceRange location)
{
	return loc(location.getStartPosition());
}

llvm::Optional<mlir::ModuleOp> MLIRLowerer::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (const auto& cls : classes)
	{
		auto* op = cls->visit([&](const auto& obj) {
			return lower(obj);
		});

		if (op != nullptr)
			operations.push_back(op);
	}

	if (operations.size() == 1 && mlir::isa<mlir::ModuleOp>(operations[0]))
		module = mlir::cast<mlir::ModuleOp>(operations[0]);
	else
	{
		module = mlir::ModuleOp::create(builder.getUnknownLoc());

		for (const auto& op : operations)
			module.push_back(op);
	}

	if (failed(mlir::verify(module)))
		return llvm::None;

	return module;
}

mlir::Operation* MLIRLowerer::lower(const frontend::Class& cls)
{
	return cls.visit([&](const auto& obj) {
		return lower(obj);
	});
}

mlir::Operation* MLIRLowerer::lower(const frontend::PartialDerFunction& function)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	auto location = loc(function.getLocation());

	llvm::StringRef derivedFunction = function.getDerivedFunction()->get<ReferenceAccess>()->getName();
	llvm::SmallVector<llvm::StringRef, 3> independentVariables;

	for (const auto& independentVariable : function.getIndependentVariables())
		independentVariables.push_back(independentVariable->get<ReferenceAccess>()->getName());

	return builder.create<DerFunctionOp>(location, function.getName(), derivedFunction, independentVariables);
}

mlir::Operation* MLIRLowerer::lower(const frontend::StandardFunction& function)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

	auto location = loc(function.getLocation());

	llvm::SmallVector<llvm::StringRef, 3> argNames;
	llvm::SmallVector<mlir::Type, 3> argTypes;

	for (const auto& member : function.getArgs())
	{
		argNames.emplace_back(member->getName());

		mlir::Type type = lower(member->getType(), BufferAllocationScope::unknown);

		if (auto arrayType = type.dyn_cast<ArrayType>())
			type = arrayType.toUnknownAllocationScope();

		argTypes.emplace_back(type);
	}

	llvm::SmallVector<llvm::StringRef, 3> returnNames;
	llvm::SmallVector<mlir::Type, 3> returnTypes;
	auto outputMembers = function.getResults();

	for (const auto& member : outputMembers)
	{
		const auto& frontendType = member->getType();
		mlir::Type type = lower(member->getType(), BufferAllocationScope::heap);

		if (auto arrayType = type.dyn_cast<ArrayType>())
			type = arrayType.toAllocationScope(BufferAllocationScope::heap);

		returnNames.emplace_back(member->getName());
		returnTypes.emplace_back(type);
	}

	auto functionType = builder.getFunctionType(argTypes, returnTypes);
	auto functionOp = builder.create<FunctionOp>(location, function.getName(), functionType, argNames, returnNames);

	if (function.hasAnnotation())
	{
		const auto* annotation = function.getAnnotation();

		// Inline attribute
		functionOp->setAttr("inline", builder.getBoolAttr(function.getAnnotation()->getInlineProperty()));

		{
			// Inverse functions attribute
			auto inverseFunctionAnnotation = annotation->getInverseFunctionAnnotation();
			InverseFunctionsAttribute::Map map;

			// Create a map of the function members indexes for faster retrieval
			llvm::StringMap<unsigned int> indexes;

			for (const auto& name : llvm::enumerate(argNames))
				indexes[name.value()] = name.index();

			for (const auto& name : llvm::enumerate(returnNames))
				indexes[name.value()] = argNames.size() + name.index();

			mlir::StorageUniquer::StorageAllocator allocator;

			// Iterate over the input arguments and for each invertible one
			// add the function to the inverse map.
			for (const auto& arg : argNames)
			{
				if (!inverseFunctionAnnotation.isInvertible(arg))
					continue;

				auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
				llvm::SmallVector<unsigned int, 3> permutation;

				for (const auto& inverseArg : inverseArgs)
				{
					assert(indexes.find(inverseArg) != indexes.end());
					permutation.push_back(indexes[inverseArg]);
				}

				map[indexes[arg]] = std::make_pair(
						inverseFunctionAnnotation.getInverseFunction(arg),
						allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
			}

			if (!map.empty())
			{
				auto inverseFunctionAttribute = builder.getInverseFunctionsAttribute(map);
				functionOp->setAttr("inverse", inverseFunctionAttribute);
			}
		}

		if (annotation->hasDerivativeAnnotation())
		{
			auto derivativeAnnotation = annotation->getDerivativeAnnotation();
			auto derivativeAttribute = builder.getDerivativeAttribute(derivativeAnnotation.getName(), derivativeAnnotation.getOrder());
			functionOp->setAttr("derivative", derivativeAttribute);
		}
	}

	// If the function doesn't have a body, it means it is just a declaration
	if (function.getAlgorithms().empty())
		return { functionOp };

	// Start the body of the function.
	auto& entryBlock = *functionOp.addEntryBlock();

	// Declare all the function arguments in the symbol table
	for (const auto& [name, value] : llvm::zip(argNames, entryBlock.getArguments()))
		symbolTable.insert(name, Reference::ssa(&builder, value));

	builder.setInsertionPointToStart(&entryBlock);

	// Initialize members
	for (const auto& member : function.getMembers())
		lower<Function>(*member);

	// Emit the body of the function
	const auto& algorithm = function.getAlgorithms()[0];

	// Lower the statements
	lower(*function.getAlgorithms()[0]);

	builder.create<FunctionTerminatorOp>(location);
	return functionOp;
}

mlir::Operation* MLIRLowerer::lower(const frontend::Model& model)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

	auto location = loc(model.getLocation());

	llvm::SmallVector<mlir::Type, 3> args;

	// Time variable
	args.push_back(builder.getArrayType(BufferAllocationScope::unknown, builder.getRealType()));

	// Other variables
	llvm::SmallVector<mlir::Attribute, 3> variableNames;

	for (const auto& member : model.getMembers())
	{
		mlir::Type type = lower(member->getType(), BufferAllocationScope::unknown);

		if (!type.isa<ArrayType>())
			type = builder.getArrayType(BufferAllocationScope::unknown, type);

		args.push_back(type);

		mlir::StringAttr nameAttribute = builder.getStringAttr(member->getName());
		variableNames.push_back(nameAttribute);
	}

	llvm::ArrayRef<mlir::Attribute> attributeArray(variableNames);
	mlir::ArrayAttr variableNamesAttribute = builder.getArrayAttr(attributeArray);

	// Create the operation
	auto simulation = builder.create<SimulationOp>(
			location,
			variableNamesAttribute,
			builder.getRealAttribute(options.startTime),
			builder.getRealAttribute(options.endTime),
			builder.getRealAttribute(options.timeStep),
			args);

	{
		// Simulation variables
		builder.setInsertionPointToStart(&simulation.init().front());
		llvm::SmallVector<mlir::Value, 3> vars;

		mlir::Value time = builder.create<AllocOp>(location, builder.getRealType(), llvm::None, llvm::None, false);
		vars.push_back(time);

		for (const auto& member : model.getMembers())
		{
			lower<frontend::Model>(*member);
			vars.push_back(symbolTable.lookup(member->getName()).getReference());
		}

		builder.create<YieldOp>(location, vars);
	}

	{
		// Body
		builder.setInsertionPointToStart(&simulation.body().front());

		mlir::Value time = simulation.time();
		symbolTable.insert("time", Reference::memory(&builder, time));

		for (const auto& member : llvm::enumerate(model.getMembers()))
			symbolTable.insert(member.value()->getName(), Reference::memory(&builder, simulation.body().getArgument(member.index() + 1)));

		for (const auto& equation : model.getEquations())
			lower(*equation);

		for (const auto& forEquation : model.getForEquations())
			lower(*forEquation);

		builder.create<YieldOp>(location);
	}

	return simulation;
}

mlir::Operation* MLIRLowerer::lower(const frontend::Package& package)
{
	mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

	for (const auto& innerClass : package.getInnerClasses())
	{
		auto* op = innerClass->visit([&](const auto& obj) {
			return lower(obj);
		});

		if (op != nullptr)
			module.push_back(op);
	}

	return module;
}

mlir::Operation* MLIRLowerer::lower(const Record& record)
{
	/*
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
	auto location = loc(record.getLocation());

	// Whenever a record is defined, a record constructor function with the
	// same name and in the same scope as the record class must be implicitly
	// defined, so that the record can then be instantiated.

	llvm::SmallVector<mlir::Type, 3> argsTypes;
	llvm::SmallVector<mlir::Type, 3> recordTypes;

	for (const auto& member : record)
	{
		argsTypes.push_back(lower(member.getType(), BufferAllocationScope::unknown));
		recordTypes.push_back(lower(member.getType(), BufferAllocationScope::heap));
	}

	RecordType resultType = builder.getRecordType(recordTypes);

	auto functionType = builder.getFunctionType(argsTypes, resultType);
	auto function = mlir::FuncOp::create(location, record.getName(), functionType);

	auto& entryBlock = *function.addEntryBlock();
	builder.setInsertionPointToStart(&entryBlock);

	llvm::SmallVector<mlir::Value, 3> results;

	for (const auto& [arg, type] : llvm::zip(entryBlock.getArguments(), recordTypes))
	{
		if (auto arrayType = type.dyn_cast<ArrayType>())
			results.push_back(builder.create<ArrayCloneOp>(location, arg, arrayType, false));
		else
			results.push_back(arg);
	}

	mlir::Value result = builder.create<RecordOp>(location, resultType, results);
	builder.create<mlir::ReturnOp>(location, result);

	return { function };
	 */

	return nullptr;
}

mlir::Type MLIRLowerer::lower(const Type& type, BufferAllocationScope desiredAllocationScope)
{
	auto visitor = [&](const auto& obj) -> mlir::Type
	{
		auto baseType = lower(obj, desiredAllocationScope);

		if (!type.isScalar())
		{
			const auto& dimensions = type.getDimensions();
			llvm::SmallVector<long, 3> shape;

			for (const auto& dimension : type.getDimensions())
			{
				if (dimension.isDynamic())
					shape.emplace_back(-1);
				else
					shape.emplace_back(dimension.getNumericSize());
			}

			return builder.getArrayType(desiredAllocationScope, baseType, shape).toMinAllowedAllocationScope();
		}

		return baseType;
	};

	return type.visit(visitor);
}

mlir::Type MLIRLowerer::lower(const BuiltInType& type, BufferAllocationScope desiredAllocationScope)
{
	switch (type)
	{
		case BuiltInType::None:
			return builder.getNoneType();
		case BuiltInType::Integer:
			return builder.getIntegerType();
		case BuiltInType::Float:
			return builder.getRealType();
		case BuiltInType::Boolean:
			return builder.getBooleanType();
		default:
			assert(false && "Unexpected type");
			return builder.getNoneType();
	}
}

mlir::Type MLIRLowerer::lower(const PackedType& type, BufferAllocationScope desiredAllocationScope)
{
	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& subType : type)
		types.push_back(lower(subType, desiredAllocationScope));

	return builder.getTupleType(move(types));
}

mlir::Type MLIRLowerer::lower(const UserDefinedType& type, BufferAllocationScope desiredAllocationScope)
{
	llvm::SmallVector<mlir::Type, 3> types;

	for (const auto& subType : type)
		types.push_back(lower(subType, desiredAllocationScope));

	return builder.getTupleType(move(types));
}

template<>
void MLIRLowerer::lower<frontend::Model>(const Member& member)
{
	auto location = loc(member.getLocation());

	const auto& frontendType = member.getType();
	mlir::Type type = lower(frontendType, BufferAllocationScope::heap);

	if (auto arrayType = type.dyn_cast<ArrayType>())
	{
		mlir::Value ptr = builder.create<AllocOp>(location, arrayType.getElementType(), arrayType.getShape(), llvm::None, false, member.isParameter());
		symbolTable.insert(member.getName(), Reference::memory(&builder, ptr));
	}
	else
	{
		mlir::Value ptr = builder.create<AllocOp>(location, type, llvm::None, llvm::None, false, member.isParameter());
		symbolTable.insert(member.getName(), Reference::memory(&builder, ptr));
	}

	mlir::Value destination = symbolTable.lookup(member.getName()).getReference();
	bool isConstant = member.isParameter();

	if (member.hasStartOverload())
	{
		auto values = lower<Expression>(*member.getStartOverload());
		assert(values.size() == 1);

		if (auto arrayType = type.dyn_cast<ArrayType>())
			builder.create<FillOp>(location, *values[0], destination);
		else
			builder.create<AssignmentOp>(location, *values[0], destination);
	}
	else if (member.hasInitializer())
	{
		Reference memory = symbolTable.lookup(member.getName());
		mlir::Value value = *lower<Expression>(*member.getInitializer())[0];
		memory.set(value);
	}
	else
	{
		if (auto arrayType = type.dyn_cast<ArrayType>())
		{
			mlir::Value zero = builder.create<ConstantOp>(location, builder.getZeroAttribute(arrayType.getElementType()));
			builder.create<FillOp>(location, zero, destination);
		}
		else
		{
			mlir::Value zero = builder.create<ConstantOp>(location, builder.getZeroAttribute(type));
			builder.create<AssignmentOp>(location, zero, destination);
		}
	}
}

/**
 * Lower a member of a function.
 *
 * Input members are ignored because they are supposed to be unmodifiable
 * as per the Modelica standard, and thus don't need a local copy.
 * Output arrays are always allocated on the heap and eventually moved to
 * input arguments by the dedicated pass. Protected arrays, instead, are
 * allocated according to the ArrayType allocation logic.
 */
template<>
void MLIRLowerer::lower<Function>(const Member& member)
{
	auto location = loc(member.getLocation());

	// Input values are supposed to be read-only by the Modelica standard,
	// thus they don't need to be copied for local modifications.

	if (member.isInput())
		return;

	const auto& frontendType = member.getType();
	mlir::Type type = lower(frontendType, member.isOutput() ? BufferAllocationScope::heap : BufferAllocationScope::stack);

	llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
	MemberType::Shape shape;

	if (auto arrayType = type.dyn_cast<ArrayType>())
	{
		for (auto dimension : arrayType.getShape())
			shape.push_back(dimension);

		auto expressionsCount = llvm::count_if(
				member.getType().getDimensions(),
				[](const auto& dimension) {
					return dimension.hasExpression();
				});

		// If all the dynamic dimensions have an expression to determine their
		// values, then the member can be instantiated from the beginning.

		bool initialized = expressionsCount == arrayType.getDynamicDimensions();

		if (initialized)
		{
			for (const auto& dimension : member.getType().getDimensions())
			{
				if (dimension.hasExpression())
				{
					mlir::Value size = *lower<Expression>(*dimension.getExpression())[0];
					size = builder.create<CastOp>(location, size, builder.getIndexType());
					dynamicDimensions.push_back(size);
				}
			}
		}
	}

	auto memberType = type.isa<ArrayType>() ?
	    MemberType::get(type.cast<ArrayType>()) :
			MemberType::get(builder.getContext(), MemberAllocationScope::stack, type);

	mlir::Value var = builder.create<MemberCreateOp>(location, member.getName(), memberType, dynamicDimensions);
	symbolTable.insert(member.getName(), Reference::member(&builder, var));

	if (member.hasInitializer())
	{
		// If the member has an initializer expression, lower and assign it as
		// if it was a regular assignment statement.

		Reference memory = symbolTable.lookup(member.getName());
		mlir::Value value = *lower<Expression>(*member.getInitializer())[0];
		memory.set(value);
	}
}

void MLIRLowerer::lower(const Equation& equation)
{
	mlir::Location location = loc(equation.getLocation());
	auto result = builder.create<EquationOp>(location);
	mlir::OpBuilder::InsertionGuard guard(builder);
	builder.setInsertionPointToStart(result.body());

	llvm::SmallVector<mlir::Value, 1> lhs;
	llvm::SmallVector<mlir::Value, 1> rhs;

	{
		// Left-hand side
		const auto* expression = equation.getLhsExpression();
		auto references = lower<Expression>(*expression);

		for (auto& reference : references)
			lhs.push_back(*reference);
	}

	{
		// Right-hand side
		const auto* expression = equation.getRhsExpression();
		auto references = lower<Expression>(*expression);

		for (auto& reference : references)
			rhs.push_back(*reference);
	}

	builder.create<EquationSidesOp>(location, lhs, rhs);
}

void MLIRLowerer::lower(const ForEquation& forEquation)
{
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
	mlir::Location location = loc(forEquation.getEquation()->getLocation());

	auto result = builder.create<ForEquationOp>(location, forEquation.getInductions().size());

	{
		// Inductions
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(result.inductionsBlock());
		llvm::SmallVector<mlir::Value, 3> inductions;

		for (const auto& induction : forEquation.getInductions())
		{
			const auto& startExpression = induction->getBegin();
			assert(startExpression->isa<Constant>());
			long start = startExpression->get<Constant>()->as<BuiltInType::Integer>();

			const auto& endExpression = induction->getEnd();
			assert(endExpression->isa<Constant>());
			long end = endExpression->get<Constant>()->as<BuiltInType::Integer>();

			mlir::Value ind = builder.create<InductionOp>(location, start, end);
			inductions.push_back(ind);
		}

		builder.create<YieldOp>(location, inductions);
	}

	{
		// Body
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPointToStart(result.body());

		// Add the induction variables to the symbol table
		for (auto [induction, var] : llvm::zip(forEquation.getInductions(), result.inductions()))
			symbolTable.insert(induction->getName(), Reference::ssa(&builder, var));

		const auto& equation = forEquation.getEquation();

		llvm::SmallVector<mlir::Value, 1> lhs;
		llvm::SmallVector<mlir::Value, 1> rhs;

		{
			// Left-hand side
			const auto* expression = equation->getLhsExpression();
			auto references = lower<Expression>(*expression);

			for (auto& reference : references)
				lhs.push_back(*reference);
		}

		{
			// Right-hand side
			const auto* expression = equation->getRhsExpression();
			auto references = lower<Expression>(*expression);

			for (auto& reference : references)
				rhs.push_back(*reference);
		}

		builder.create<EquationSidesOp>(location, lhs, rhs);
	}
}

void MLIRLowerer::lower(const Algorithm& algorithm)
{
	for (const auto& statement : algorithm)
		lower(*statement);
}

void MLIRLowerer::lower(const Statement& statement)
{
	statement.visit([&](const auto& obj) { lower(obj); });
}

void MLIRLowerer::lower(const AssignmentStatement& statement)
{
	const auto* destinations = statement.getDestinations();
	auto values = lower<Expression>(*statement.getExpression());

	assert(destinations->isa<Tuple>());
	const auto* destinationsTuple = destinations->get<Tuple>();
	assert(values.size() == destinationsTuple->size() && "Unequal number of destinations and results");

	for (const auto& [ dest, value ] : llvm::zip(*destinationsTuple, values))
	{
		auto destination = lower<Expression>(*dest)[0];
		destination.set(*value);
	}
}

void MLIRLowerer::lower(const IfStatement& statement)
{
	// Each conditional blocks creates an If operation, but we need to keep
	// track of the first one in order to restore the insertion point right
	// after that when we have finished lowering all the blocks.
	mlir::Operation* firstOp = nullptr;

	size_t blocks = statement.size();

	for (size_t i = 0; i < blocks; ++i)
	{
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
		const auto& conditionalBlock = statement[i];
		auto condition = lower<Expression>(*conditionalBlock.getCondition())[0];

		// The last conditional block can be at most an originally equivalent
		// "else" block, and thus doesn't need a lowered else block.
		bool elseBlock = i < blocks - 1;

    auto location = loc(statement.getLocation());
		auto ifOp = builder.create<IfOp>(location, *condition, elseBlock);

		if (firstOp == nullptr)
			firstOp = ifOp;

		// "Then" block
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (const auto& stmnt : conditionalBlock)
			lower(*stmnt);

    if (auto& block = ifOp.thenRegion().back(); block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
      builder.create<YieldOp>(location);

		if (i > 0)
		{
			builder.setInsertionPointAfter(ifOp);
      builder.create<YieldOp>(location);
		}

		// The next conditional blocks will be placed as new If operations
		// nested inside the "else" block.
		if (elseBlock)
			builder.setInsertionPointToStart(&ifOp.elseRegion().front());
	}

	builder.setInsertionPointAfter(firstOp);
}

void MLIRLowerer::lower(const ForStatement& statement)
{
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
	auto location = loc(statement.getLocation());

	const auto& induction = statement.getInduction();

	mlir::Value lowerBound = *lower<Expression>(*induction->getBegin())[0];
	lowerBound = builder.create<CastOp>(lowerBound.getLoc(), lowerBound, builder.getIndexType());

	auto forOp = builder.create<ForOp>(location, lowerBound);
	mlir::OpBuilder::InsertionGuard guard(builder);

	{
		// Check the loop condition
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.condition().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.condition().front());

		mlir::Value upperBound = *lower<Expression>(*induction->getEnd())[0];
		upperBound = builder.create<CastOp>(lowerBound.getLoc(), upperBound, builder.getIndexType());

		mlir::Value condition = builder.create<LteOp>(location, builder.getBooleanType(), forOp.condition().front().getArgument(0), upperBound);
		builder.create<ConditionOp>(location, condition, *symbolTable.lookup(induction->getName()));
	}

	{
		// Body
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.body().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.body().front());

		for (const auto& stmnt : statement)
			lower(*stmnt);

    if (auto& body = forOp.body().back(); body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
      builder.setInsertionPointToEnd(&body);
      builder.create<YieldOp>(location, *symbolTable.lookup(induction->getName()));
    }
	}

	{
		// Step
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction->getName(), Reference::ssa(&builder, forOp.step().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.step().front());

		mlir::Value step = builder.create<ConstantOp>(location, builder.getIndexAttribute(1));
		mlir::Value incremented = builder.create<AddOp>(location, builder.getIndexType(), *symbolTable.lookup(induction->getName()), step);
		builder.create<YieldOp>(location, incremented);
	}
}

void MLIRLowerer::lower(const WhileStatement& statement)
{
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
	auto location = loc(statement.getLocation());

	// Create the operation
	auto whileOp = builder.create<WhileOp>(location);
	mlir::OpBuilder::InsertionGuard guard(builder);

	{
		// Condition
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		mlir::Block* conditionBlock = &whileOp.condition().front();
		builder.setInsertionPointToStart(conditionBlock);
		const auto* condition = statement.getCondition();

		builder.create<ConditionOp>(
				loc(condition->getLocation()),
				*lower<Expression>(*condition)[0]);
	}

	{
		// Body
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		builder.setInsertionPointToStart(&whileOp.body().front());

		for (const auto& stmnt : statement)
			lower(*stmnt);

    if (auto& body = whileOp.body().back(); body.empty() || !body.back().hasTrait<mlir::OpTrait::IsTerminator>())
    {
      builder.setInsertionPointToEnd(&body);
      builder.create<YieldOp>(location);
    }
	}
}

void MLIRLowerer::lower(const WhenStatement& statement)
{

}

void MLIRLowerer::lower(const BreakStatement& statement)
{
	mlir::Location location = loc(statement.getLocation());
	builder.create<BreakOp>(location);
}

void MLIRLowerer::lower(const ReturnStatement& statement)
{
	mlir::Location location = loc(statement.getLocation());
	builder.create<ReturnOp>(location);
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Expression>(const Expression& expression)
{
	return expression.visit([&](const auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return lower<deconst>(expression);
	});
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Operation>(const Expression& expression)
{
	assert(expression.isa<Operation>());
	const auto* operation = expression.get<Operation>();
	auto kind = operation->getOperationKind();
	mlir::Location location = loc(expression.getLocation());

	if (kind == OperationKind::negate)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto arg = lower<Expression>(*operation->getArg(0))[0].getReference();
		mlir::Value result = builder.create<NotOp>(location, resultType, arg);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::add)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);

		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<AddOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subtract)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);

		if (args.size() == 1)
		{
			// Special case for sign change (i.e "-x").
			// TODO
			// In future, when all the project will rely on MLIR, a different
			// operation in the frontend should be created for this purpose.

			mlir::Value result = builder.create<NegateOp>(location, resultType, args[0]);
			return { Reference::ssa(&builder, result) };
		}

		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<SubOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::multiply)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);

		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<MulOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::divide)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);

		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<DivOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::powerOf)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		mlir::Value base = *lower<Expression>(*operation->getArg(0))[0];
		mlir::Value exponent = *lower<Expression>(*operation->getArg(1))[0];

		/*
		if (base.getType().isa<ArrayType>())
		{
			exponent = builder.create<CastOp>(base.getLoc(), exponent, builder.getIntegerType());
		}
		else
		{
			base = builder.create<CastOp>(base.getLoc(), base, builder.getRealType());
			exponent = builder.create<CastOp>(base.getLoc(), exponent, builder.getRealType());
		}
		 */

		mlir::Value result = builder.create<PowOp>(location, resultType, base, exponent);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::equal)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<EqOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::different)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<NotEqOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greater)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<GtOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greaterEqual)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<GteOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::less)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<LtOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::lessEqual)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		auto args = lowerOperationArgs(*operation);
		mlir::Value result = builder.create<LteOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::ifelse)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		mlir::Value condition = *lower<Expression>(*operation->getArg(0))[0];

		mlir::Value trueValue = *lower<Expression>(*operation->getArg(1))[0];
		trueValue = builder.create<CastOp>(trueValue.getLoc(), trueValue, resultType);

		mlir::Value falseValue = *lower<Expression>(*operation->getArg(2))[0];
		falseValue = builder.create<CastOp>(falseValue.getLoc(), falseValue, resultType);

		mlir::Value result = builder.create<mlir::SelectOp>(location, condition, trueValue, falseValue);
		result = builder.create<CastOp>(result.getLoc(), result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::land)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		mlir::Value lhs = *lower<Expression>(*operation->getArg(0))[0];
		mlir::Value rhs = *lower<Expression>(*operation->getArg(1))[0];

		mlir::Value result = builder.create<AndOp>(location, resultType, lhs, rhs);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::lor)
	{
		mlir::Type resultType = lower(operation->getType(), BufferAllocationScope::stack);
		mlir::Value lhs = *lower<Expression>(*operation->getArg(0))[0];
		mlir::Value rhs = *lower<Expression>(*operation->getArg(1))[0];

		mlir::Value result = builder.create<OrOp>(location, resultType, lhs, rhs);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subscription)
	{
		auto buffer = *lower<Expression>(*operation->getArg(0))[0];
		assert(buffer.getType().isa<ArrayType>());

		llvm::SmallVector<mlir::Value, 3> indexes;

		for (size_t i = 1; i < operation->argumentsCount(); i++)
		{
			mlir::Value index = *lower<Expression>(*operation->getArg(i))[0];
			indexes.push_back(index);
		}

		mlir::Value result = builder.create<SubscriptionOp>(location, buffer, indexes);
		return { Reference::memory(&builder, result) };
	}

	if (kind == OperationKind::memberLookup)
	{
		// TODO
		return { Reference::ssa(&builder, nullptr) };
	}

	assert(false && "Unexpected operation");
	return { Reference::ssa(&builder, mlir::Value()) };
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Constant>(const Expression& expression)
{
	assert(expression.isa<Constant>());
	const auto* constant = expression.get<Constant>();
	const auto& type = constant->getType();

	assert(
			type.isa<BuiltInType>() && "Constants can be made only of built-in typed values");
	auto builtInType = type.get<BuiltInType>();

	mlir::Attribute attribute;

	if (builtInType == BuiltInType::Boolean)
		attribute = builder.getBooleanAttribute(constant->as<BuiltInType::Boolean>());
	else if (builtInType == BuiltInType::Integer)
		attribute = builder.getIntegerAttribute(constant->as<BuiltInType::Integer>());
	else if (builtInType == BuiltInType::Float)
		attribute = builder.getRealAttribute(constant->as<BuiltInType::Float>());
	else
		assert(false && "Unsupported constant type");

	auto result = builder.create<ConstantOp>(loc(expression.getLocation()), attribute);
	return { Reference::ssa(&builder, result) };
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<ReferenceAccess>(const Expression& expression)
{
	assert(expression.isa<ReferenceAccess>());
	const auto& reference = expression.get<ReferenceAccess>();
	return { symbolTable.lookup(reference->getName()) };
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Call>(const Expression& expression)
{
	assert(expression.isa<Call>());
	const auto* call = expression.get<Call>();
	const auto* function = call->getFunction();
	mlir::Location location = loc(expression.getLocation());

	const auto& functionName = function->get<ReferenceAccess>()->getName();

	Container<Reference> results;

	if (functionName == "abs")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<AbsOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "acos")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<AcosOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "asin")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<AsinOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "atan")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<AtanOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "atan2")
	{
		assert(call->argumentsCount() == 2);
		mlir::Value y = *lower<Expression>(*call->getArg(0))[0];
		mlir::Value x = *lower<Expression>(*call->getArg(1))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<Atan2Op>(location, resultType, y, x);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "cos")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<CosOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "cosh")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<CoshOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "der")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = lower<Expression>(*call->getArg(0))[0].getReference();
		assert(operand.getType().isa<ArrayType>());
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<DerOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "diagonal")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value values = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<DiagonalOp>(location, resultType, values);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "exp")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value exponent = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<ExpOp>(location, resultType, exponent);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "identity")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value size = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<IdentityOp>(location, resultType, size);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "linspace")
	{
		assert(call->argumentsCount() == 3);
		mlir::Value start = *lower<Expression>(*call->getArg(0))[0];
		mlir::Value end = *lower<Expression>(*call->getArg(1))[0];
		mlir::Value steps = *lower<Expression>(*call->getArg(2))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<LinspaceOp>(location, resultType, start, end, steps);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "log")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<LogOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "log10")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<Log10Op>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "max")
	{
		// The min function can have either one array operand or two scalar operands
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
			args.push_back(*lower<Expression>(*arg)[0]);

		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<MaxOp>(location, resultType, args);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "min")
	{
		// The min function can have either one array operand or two scalar operands
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
			args.push_back(*lower<Expression>(*arg)[0]);

		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<MinOp>(location, resultType, args);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "ndims")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value memory = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<NDimsOp>(location, resultType, memory);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "ones")
	{
		// The number of operands is equal to the rank of the resulting array
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
			args.push_back(*lower<Expression>(*arg)[0]);

		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<OnesOp>(location, resultType, args);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "product")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value memory = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<ProductOp>(location, resultType, memory);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "sign")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SignOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "sin")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SinOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "sinh")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SinhOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "size")
	{
		assert(call->argumentsCount() == 1 || call->argumentsCount() == 2);
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
			args.push_back(*lower<Expression>(*arg)[0]);

		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);

		if (args.size() == 1)
		{
			mlir::Value result = builder.create<SizeOp>(location, resultType, args[0]);
			results.emplace_back(Reference::ssa(&builder, result));
		}
		else if (args.size() == 2)
		{
			mlir::Value oneValue = builder.create<ConstantOp>(location, builder.getIntegerAttribute(1));
			mlir::Value index = builder.create<SubOp>(location, builder.getIndexType(), args[1], oneValue);
			mlir::Value result = builder.create<SizeOp>(location, resultType, args[0], index);
			results.emplace_back(Reference::ssa(&builder, result));
		}
	}
	else if (functionName == "sqrt")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SqrtOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "sum")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value memory = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SumOp>(location, resultType, memory);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "symmetric")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value source = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<SymmetricOp>(location, resultType, source);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "tan")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<TanOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "tanh")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value operand = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<TanhOp>(location, resultType, operand);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "transpose")
	{
		assert(call->argumentsCount() == 1);
		mlir::Value memory = *lower<Expression>(*call->getArg(0))[0];
		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<TransposeOp>(location, resultType, memory);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else if (functionName == "zeros")
	{
		// The number of operands is equal to the rank of the resulting array
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
			args.push_back(*lower<Expression>(*arg)[0]);

		mlir::Type resultType = lower(call->getType(), BufferAllocationScope::stack);
		mlir::Value result = builder.create<ZerosOp>(location, resultType, args);
		results.emplace_back(Reference::ssa(&builder, result));
	}
	else
	{
		llvm::SmallVector<mlir::Value, 3> args;

		for (const auto& arg : *call)
		{
			auto reference = lower<Expression>(*arg)[0];
			args.push_back(*reference);
		}

		auto resultType = call->getType();
		llvm::SmallVector<mlir::Type, 3> callResultsTypes;

		if (resultType.isa<PackedType>())
		{
			for (const auto& type : resultType.get<PackedType>())
				callResultsTypes.push_back(lower(type, BufferAllocationScope::heap));
		}
		else
			callResultsTypes.push_back(lower(resultType, BufferAllocationScope::heap));

		auto op = builder.create<CallOp>(
				loc(expression.getLocation()),
				function->get<ReferenceAccess>()->getName(),
				callResultsTypes,
				args);

		for (auto result : op->getResults())
			results.push_back(Reference::ssa(&builder, result));
	}

	return results;
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Tuple>(const Expression& expression)
{
	assert(expression.isa<Tuple>());
	const auto* tuple = expression.get<Tuple>();
	Container<Reference> result;

	for (const auto& exp : *tuple)
	{
		auto values = lower<Expression>(expression);

		// The only way to have multiple returns is to call a function, but this
		// is forbidden in a tuple declaration. In fact, a tuple is just a
		// container of references.
		assert(values.size() == 1);
		result.emplace_back(values[0]);
	}

	return result;
}

template<>
MLIRLowerer::Container<Reference> MLIRLowerer::lower<Array>(const Expression& expression)
{
	assert(expression.isa<Array>());
	const auto& array = expression.get<Array>();
	mlir::Location location = loc(expression.getLocation());
	auto type = lower(array->getType(), BufferAllocationScope::stack).cast<ArrayType>();

	mlir::Value result = builder.create<AllocaOp>(location, type.getElementType(), type.getShape());

	for (const auto& value : llvm::enumerate(*array))
	{
		mlir::Value index = builder.create<ConstantOp>(location, builder.getIndexAttribute(value.index()));
		mlir::Value slice = builder.create<SubscriptionOp>(location, result, index);
		builder.create<AssignmentOp>(location, *lower<Expression>(*value.value())[0], slice);
	}

	return { Reference::ssa(&builder, result) };
}

mlir::Value MLIRLowerer::foldBinaryOperation(llvm::ArrayRef<mlir::Value> args, std::function<mlir::Value(mlir::Value, mlir::Value)> callback)
{
	assert(args.size() >= 2);
	mlir::Value result = callback(args[0], args[1]);

	for (size_t i = 2, e = args.size(); i < e; ++i)
		result = callback(result, args[i]);

	return result;
}

MLIRLowerer::Container<mlir::Value> MLIRLowerer::lowerOperationArgs(const Operation& operation)
{
	Container<mlir::Value> args;

	for (const auto& arg : operation)
		args.push_back(*lower<Expression>(*arg)[0]);

	return args;
}
