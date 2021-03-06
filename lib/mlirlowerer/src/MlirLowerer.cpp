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
#include <modelica/mlirlowerer/passes/LowerToLLVM.h>
#include <modelica/mlirlowerer/passes/LowerModelica.h>
#include <modelica/mlirlowerer/MlirLowerer.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica;
using namespace std;

mlir::LogicalResult modelica::convertToLLVMDialect(mlir::MLIRContext* context, mlir::ModuleOp module, ModelicaOptions options)
{
	mlir::PassManager passManager(context);

	// Lower the Modelica dialect
	//passManager.addPass(createModelicaLoweringPass());

	// Lower the Affine dialect
	//passManager.addPass(createLowerAffinePass());

	// Convert the output buffers to input buffers, in order to delegate the
	// buffer allocation to the caller.
	//passManager.addPass(createBufferResultsToOutParamsPass());

	// Convert vector operations to loops
	//passManager.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

	// Lower the SCF operations
	//passManager.addPass(createLowerToCFGPass());

	//passManager.addNestedPass<FuncOp>(createMemRefDataFlowOptPass());

	// Conversion to LLVM dialect
	ModelicaToLLVMLoweringOptions modelicaToLLVMOptions;
	passManager.addPass(createModelicaToLLVMLoweringPass(modelicaToLLVMOptions));

	passManager.addPass(mlir::createConvertVectorToLLVMPass());

	return passManager.run(module);
}

Reference::Reference() : builder(nullptr), value(nullptr), reader(nullptr)
{
}

Reference::Reference(ModelicaBuilder* builder,
										 mlir::Value value,
										 bool initialized,
										 std::function<mlir::Value(ModelicaBuilder*, mlir::Value)> reader)
		: builder(builder),
			value(std::move(value)),
			initialized(initialized),
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

bool Reference::isInitialized() const
{
	return initialized;
}

Reference Reference::ssa(ModelicaBuilder* builder, mlir::Value value)
{
	return Reference(
			builder, value, true,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value
			{
				return value;
			});
}

Reference Reference::memref(ModelicaBuilder* builder, mlir::Value value, bool initialized)
{
	return Reference(
			builder, value, initialized,
			[](mlir::OpBuilder* builder, mlir::Value value) -> mlir::Value
			{
				return builder->create<modelica::LoadOp>(value.getLoc(), value);
			});
}

MlirLowerer::MlirLowerer(mlir::MLIRContext& context, ModelicaOptions options)
		: builder(&context, options.getBitWidth()), options(move(options))
{
	context.loadDialect<ModelicaDialect>();
	context.loadDialect<mlir::StandardOpsDialect>();
	context.loadDialect<mlir::scf::SCFDialect>();
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

	if (source.isa<mlir::ShapedType>())
	{
		sourceBase = source.cast<mlir::ShapedType>().getElementType();
		auto sourceShape = source.cast<mlir::ShapedType>().getShape();

		if (destination.isa<mlir::ShapedType>())
		{
			auto destinationShape = destination.cast<mlir::ShapedType>().getShape();
			destinationBase = destination.cast<mlir::ShapedType>().getElementType();
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
		if (destinationBase.isa<mlir::FloatType>())
			return builder.create<mlir::SIToFPOp>(value.getLoc(), value, destination);

		if (destinationBase.isIndex())
			return builder.create<mlir::IndexCastOp>(value.getLoc(), value, destination);
	}
	else if (sourceBase.isa<mlir::FloatType>())
	{
		if (destinationBase.isSignlessInteger())
			return builder.create<mlir::FPToSIOp>(value.getLoc(), value, destination);
	}
	else if (sourceBase.isIndex())
	{
		if (destinationBase.isSignlessInteger())
			return builder.create<mlir::IndexCastOp>(value.getLoc(), value, builder.getIntegerType());

		if (destinationBase.isa<mlir::FloatType>())
			return cast(builder.create<mlir::IndexCastOp>(value.getLoc(), value, builder.getIntegerType()), destination);
	}

	assert(false && "Unsupported type conversion");
	return nullptr;
}

mlir::ModuleOp MlirLowerer::lower(llvm::ArrayRef<const modelica::ClassContainer> classes)
{
	mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

	/*
	auto funcType = builder.getFunctionType(TypeRange(), TypeRange());
	auto testFoo = FuncOp::create(builder.getUnknownLoc(), "test", funcType);
	testFoo.setPrivate();
	module.push_back(testFoo);
	 */

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
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

	auto location = loc(foo.getLocation());

	llvm::SmallVector<llvm::StringRef, 3> argNames;
	llvm::SmallVector<mlir::Type, 3> argTypes;

	for (const auto& member : foo.getArgs())
	{
		argNames.emplace_back(member->getName());
		argTypes.emplace_back(lower(member->getType()));
	}

	llvm::SmallVector<llvm::StringRef, 3> returnNames;
	llvm::SmallVector<mlir::Type, 3> returnTypes;
	auto outputMembers = foo.getResults();

	for (const auto& member : outputMembers)
	{
		mlir::Type type = lower(member->getType());
		returnNames.emplace_back(member->getName());

		if (member->isOutput() && type.isa<PointerType>())
			type = builder.getPointerType(true, type.cast<PointerType>().getElementType(), type.cast<PointerType>().getShape());

		returnTypes.emplace_back(type);
	}

	auto functionType = builder.getFunctionType(argTypes, returnTypes);
	auto function = mlir::FuncOp::create(location, foo.getName(), functionType);

	// If the function doesn't have a body, it means it is just a declaration
	if (foo.getAlgorithms().empty())
		return function;

	// Start the body of the function.
	// In MLIR the entry block of the function is special: it must have the same
	// argument list as the function itself.
	auto &entryBlock = *function.addEntryBlock();

	// Declare all the function arguments in the symbol table
	for (const auto& pair : llvm::zip(argNames, entryBlock.getArguments())) {
		const auto& name = get<0>(pair);
		const auto& value = get<1>(pair);
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
	mlir::Value returnCondition = builder.create<AllocaOp>(algorithmLocation,  builder.getBooleanType());
	mlir::Value falseValue = builder.create<mlir::ConstantOp>(algorithmLocation, builder.getBooleanAttribute(false));
	builder.create<StoreOp>(algorithmLocation, falseValue, returnCondition);
	symbolTable.insert(algorithm.getReturnCheckName(), Reference::memref(&builder, returnCondition, true));



	//mlir::Value mem = *symbolTable.lookup("y");

	/*
	mlir::Value twoIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(2));
	mlir::Value twoValue = builder.create<mlir::ConstantOp>(location, builder.getIntegerAttribute(2));
	builder.create<StoreOp>(location, twoValue, mem, twoIndex);
	 */

	/*
	mlir::Value twoIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(2));
	mlir::Value twoValue = builder.create<mlir::ConstantOp>(location, builder.getIntegerType(), builder.getIntegerAttribute(2));
	mlir::Value sub2 = builder.create<SubscriptionOp>(location, mem, twoIndex);
	builder.create<AssignmentOp>(location, twoValue, sub2);
	 */



	// Lower the statements
	lower(foo.getAlgorithms()[0]);




	/*
	mlir::Value zeroIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(0));
	mlir::Value oneIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(1));
	mlir::Value twoIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(2));
	mlir::Value threeIndex = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(3));

	builder.create<mlir::vector::PrintOp>(location, builder.create<LoadOp>(location, mem, zeroIndex));
	builder.create<mlir::vector::PrintOp>(location, builder.create<LoadOp>(location, mem, oneIndex));
	builder.create<mlir::vector::PrintOp>(location, builder.create<LoadOp>(location, mem, twoIndex));
	builder.create<mlir::vector::PrintOp>(location, builder.create<LoadOp>(location, mem, threeIndex));
	 */







	// Return statement
	std::vector<mlir::Value> results;

	for (const auto& member : outputMembers)
	{
		auto ptr = symbolTable.lookup(member->getName());
		results.push_back(*ptr);
	}

	builder.create<mlir::ReturnOp>(location, results);
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
			llvm::SmallVector<long, 3> shape;

			for (const auto& dimension : type.getDimensions())
			{
				if (dimension.isDynamic())
					shape.emplace_back(-1);
				else
					shape.emplace_back(dimension.getNumericSize());
			}

			return builder.getPointerType(false, baseType, shape);
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

mlir::Type MlirLowerer::lower(const modelica::UserDefinedType& type)
{
	llvm::SmallVector<mlir::Type, 3> types;

	for (auto& subType : type)
		types.push_back(lower(subType));

	return builder.getTupleType(move(types));
}

/**
 * Lower a member of a function.
 * If the size of the element can be determined statically, then it is
 * allocated on the stack. If not, it will be allocated when it will be
 * initialized.
 * Input members are ignored because they are supposed to be unmodifiable
 * as per the Modelica standard.
 *
 * 	                 INPUT              OUTPUT               PROTECTED
 * scalar              -                stack                  stack
 * memref ranked       -                - (in param)           stack
 * memref unranked     -                heap (out param)       stack
 * @param member
 */
void MlirLowerer::lower(const modelica::Member& member)
{
	auto location = loc(member.getLocation());

	// Input values are supposed to be read-only by the Modelica standard,
	// thus they don't need to be allocated on the stack for modifications.
	if (member.isInput())
		return;

	mlir::Type type = lower(member.getType());

	if (type.isa<PointerType>())
	{
		auto pointerType = type.cast<PointerType>();

		if (member.isOutput())
			type = builder.getPointerType(true, pointerType.getElementType(), pointerType.getShape());
	}

	mlir::Value ptr = builder.create<modelica::AllocaOp>(location, type);
	bool initialized = false;

	if (type.isa<PointerType>())
	{
		auto pointerType = type.cast<PointerType>();
		llvm::SmallVector<mlir::Value, 3> sizes;

		for (const auto& dimension : member.getType().getDimensions())
			if (dimension.hasExpression())
			{
				mlir::Value size = *lower<modelica::Expression>(dimension.getExpression())[0];
				size = builder.create<CastOp>(location, size, builder.getIndexType());
				sizes.push_back(size);
			}

		if (sizes.size() == pointerType.getDynamicDimensions())
		{
			// All the dynamic dimensions have an expression to determine their values.
			// So we can instantiate the array.

			if (pointerType.isOnHeap())
			{
				mlir::Value var = builder.create<modelica::AllocOp>(location, pointerType.getElementType(), pointerType.getShape(), sizes);
				builder.create<modelica::StoreOp>(location, var, ptr);
			}
			else
			{
				mlir::Value var = builder.create<modelica::AllocaOp>(location, pointerType.getElementType(), pointerType.getShape(), sizes);
				builder.create<modelica::StoreOp>(location, var, ptr);
			}

			initialized = true;
		}
	}

	symbolTable.insert(member.getName(), Reference::memref(&builder, ptr, initialized));

	/*
	if (member.hasInitializer())
	{
		mlir::Value reference = symbolTable.lookup(member.getName()).getReference();
		mlir::Value value = lower<modelica::Expression>(member.getInitializer())[0].getReference();
		builder.create<AssignmentOp>(loc(member.getInitializer().getLocation()), value, reference);
	}
	 */
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

		auto destinationPointer = destination.getReference().getType().cast<PointerType>();

		if (destinationPointer.getElementType().isa<PointerType>())
		{
			if (destination.isInitialized())
				builder.create<AssignmentOp>(location, *value, *destination);
			else
			{
				auto pointer = destinationPointer.getElementType().cast<PointerType>();

				// Copy source on stack
				// Save the descriptor of the new copy into the destination using StoreOp

				mlir::Value copy = builder.create<ArrayCopyOp>(location, *value, pointer.isOnHeap());
				builder.create<StoreOp>(location, copy, destination.getReference());
			}
		}
		else
		{
			builder.create<AssignmentOp>(location, *value, destination.getReference());
		}
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
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
		const auto& conditionalBlock = statement[i];
		auto condition = lower<modelica::Expression>(conditionalBlock.getCondition())[0];

		// The last conditional block can be at most an originally equivalent
		// "else" block, and thus doesn't need a lowered else block.
		bool elseBlock = i < blocks - 1;

		auto ifOp = builder.create<mlir::scf::IfOp>(loc(statement.getLocation()), *condition, elseBlock);

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
	llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);

	auto location = loc(statement.getLocation());

	// Variable to be set when calling "break"
	mlir::Value breakCondition = builder.create<AllocaOp>(location, builder.getBooleanType());
	mlir::Value falseValue = builder.create<mlir::ConstantOp>(location, builder.getBooleanAttribute(false));
	builder.create<StoreOp>(location, falseValue, breakCondition);
	symbolTable.insert(statement.getBreakCheckName(), Reference::memref(&builder, breakCondition, true));

	// Variable to be set when calling "return"
	mlir::Value returnCondition = symbolTable.lookup(statement.getReturnCheckName()).getReference();

	const auto& induction = statement.getInduction();

	mlir::Value lowerBound = *lower<modelica::Expression>(induction.getBegin())[0];
	lowerBound = builder.create<CastOp>(lowerBound.getLoc(), lowerBound, builder.getIndexType());

	auto forOp = builder.create<ForOp>(location, breakCondition, returnCondition, lowerBound);

	{
		// Check the loop condition
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.condition().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.condition().front());

		mlir::Value upperBound = *lower<modelica::Expression>(induction.getEnd())[0];
		upperBound = builder.create<CastOp>(lowerBound.getLoc(), upperBound, builder.getIndexType());

		mlir::Value condition = builder.create<LteOp>(location, builder.getBooleanType(), forOp.condition().front().getArgument(0), upperBound);
		builder.create<ConditionOp>(location, condition, *symbolTable.lookup(induction.getName()));
	}

	{
		// Body
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.body().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.body().front());

		for (const auto& stmnt : statement)
			lower(stmnt);

		builder.create<YieldOp>(location, *symbolTable.lookup(induction.getName()));
	}

	{
		// Step
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> scope(symbolTable);
		symbolTable.insert(induction.getName(), Reference::ssa(&builder, forOp.step().front().getArgument(0)));

		builder.setInsertionPointToStart(&forOp.step().front());

		mlir::Value step = builder.create<mlir::ConstantOp>(location, builder.getIndexAttribute(1));
		mlir::Value incremented = builder.create<mlir::AddIOp>(location, *symbolTable.lookup(induction.getName()), step);
		builder.create<YieldOp>(location, incremented);
	}

	builder.setInsertionPointAfter(forOp);
}

void MlirLowerer::lower(const modelica::WhileStatement& statement)
{
	auto location = loc(statement.getLocation());

	// Variable to be set when calling "break"
	mlir::Value breakCondition = builder.create<AllocaOp>(location, builder.getBooleanType());
	mlir::Value falseValue = builder.create<mlir::ConstantOp>(location, builder.getBooleanAttribute(false));
	builder.create<StoreOp>(location, falseValue, breakCondition);
	symbolTable.insert(statement.getBreakCheckName(), Reference::memref(&builder, breakCondition, true));

	// Variable to be set when calling "return"
	mlir::Value returnCondition = symbolTable.lookup(statement.getReturnCheckName()).getReference();

	// Create the operation
	auto whileOp = builder.create<WhileOp>(location, breakCondition, returnCondition);

	{
		// Condition
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
		mlir::Block* conditionBlock = &whileOp.condition().front();
		builder.setInsertionPointToStart(conditionBlock);
		const auto& condition = statement.getCondition();

		builder.create<ConditionOp>(
				loc(condition.getLocation()),
				*lower<modelica::Expression>(condition)[0]);
	}

	{
		// Body
		llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
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
	mlir::Value trueValue = builder.create<mlir::ConstantOp>(location, builder.getBooleanAttribute(true));
	mlir::Value breakCondition = symbolTable.lookup(statement.getBreakCheckName()).getReference();
	builder.create<StoreOp>(location, trueValue, breakCondition);
}

void MlirLowerer::lower(const modelica::ReturnStatement& statement)
{
	auto location = loc(statement.getLocation());
	mlir::Value trueValue = builder.create<mlir::ConstantOp>(location, builder.getBooleanAttribute(true));
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
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::add)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<modelica::AddOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subtract)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<modelica::SubOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::multiply)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = foldBinaryOperation(
				args,
				[&](mlir::Value lhs, mlir::Value rhs) -> mlir::Value
				{
					return builder.create<modelica::MulOp>(location, resultType, lhs, rhs);
				});

		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::divide)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<modelica::DivOp>(location, resultType, args);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::powerOf)
	{
		mlir::Value base = *lower<modelica::Expression>(operation[0])[0];
		base = builder.create<CastOp>(base.getLoc(), base, builder.getRealType());

		mlir::Value exponent = *lower<modelica::Expression>(operation[1])[0];
		exponent = builder.create<CastOp>(base.getLoc(), exponent, builder.getRealType());

		//mlir::Value result = builder.create<PowFOp>(location, base, exponent);
		//result = builder.create<CastOp>(result.getLoc(), result, resultType);
		return { Reference::ssa(&builder, nullptr) };
	}

	if (kind == OperationKind::equal)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<EqOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::different)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<NotEqOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greater)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<GtOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::greaterEqual)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<GteOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::less)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<LtOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::lessEqual)
	{
		auto args = lowerOperationArgs(operation);
		mlir::Value result = builder.create<LteOp>(location, resultType, args[0], args[1]);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::ifelse)
	{
		mlir::Value condition = *lower<modelica::Expression>(operation[0])[0];

		mlir::Value trueValue = *lower<modelica::Expression>(operation[1])[0];
		trueValue = builder.create<CastOp>(trueValue.getLoc(), trueValue, resultType);

		mlir::Value falseValue = *lower<modelica::Expression>(operation[2])[0];
		falseValue = builder.create<CastOp>(falseValue.getLoc(), falseValue, resultType);

		mlir::Value result = builder.create<mlir::SelectOp>(location, condition, trueValue, falseValue);
		result = builder.create<CastOp>(result.getLoc(), result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::land)
	{
		mlir::Value lhs = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value rhs = *lower<modelica::Expression>(operation[1])[0];

		mlir::Value result = builder.create<mlir::AndOp>(location, lhs, rhs);
		result = builder.create<CastOp>(result.getLoc(), result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::lor)
	{
		mlir::Value lhs = *lower<modelica::Expression>(operation[0])[0];
		mlir::Value rhs = *lower<modelica::Expression>(operation[1])[0];

		mlir::Value result = builder.create<mlir::OrOp>(location, lhs, rhs);
		result = builder.create<CastOp>(result.getLoc(), result, resultType);
		return { Reference::ssa(&builder, result) };
	}

	if (kind == OperationKind::subscription)
	{
		auto buffer = *lower<modelica::Expression>(operation[0])[0];
		assert(buffer.getType().isa<PointerType>());

		llvm::SmallVector<mlir::Value, 3> indexes;

		for (size_t i = 1; i < operation.argumentsCount(); i++)
		{
			auto subscript = *lower<modelica::Expression>(operation[i])[0];
			mlir::Value index = builder.create<CastOp>(subscript.getLoc(), subscript, builder.getIndexType());
			indexes.push_back(index);
		}

		mlir::Value result = builder.create<SubscriptionOp>(location, buffer, indexes);
		return { Reference::memref(&builder, result, true) };
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

	auto value = builder.create<mlir::ConstantOp>(
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

	llvm::SmallVector<mlir::Value, 3> args;

	for (const auto& arg : call)
	{
		auto& reference = lower<modelica::Expression>(arg)[0];
		args.push_back(*reference);
	}

	auto op = builder.create<mlir::CallOp>(
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

mlir::Value MlirLowerer::foldBinaryOperation(llvm::ArrayRef<mlir::Value> args, std::function<mlir::Value(mlir::Value, mlir::Value)> callback)
{
	assert(args.size() >= 2);
	mlir::Value result = callback(args[0], args[1]);

	for (size_t i = 2, e = args.size(); i < e; ++i)
		result = callback(result, args[i]);

	return result;
}

MlirLowerer::Container<mlir::Value> MlirLowerer::lowerOperationArgs(const modelica::Operation& operation)
{
	llvm::SmallVector<mlir::Value, 3> args;
	llvm::SmallVector<mlir::Value, 3> castedArgs;

	bool containsInteger = false;
	bool containsFloat = false;

	for (const auto& arg : operation)
	{
		mlir::Location location = loc(arg.getLocation());
		const auto& type = arg.getType();

		// For now, we only support operation between built-in types.
		// In future, this should be extended to support struct types.
		assert(type.isA<modelica::BuiltInType>());

		mlir::Value value = *lower<modelica::Expression>(arg)[0];

		/*
		if (type.get<BuiltInType>() == BuiltInType::Integer)
			value = builder.create<CastOp>(location, value, builder.getIntegerType());
		else if (type.get<BuiltInType>() == BuiltInType::Float)
			value = builder.create<CastOp>(location, value, builder.getRealType());
		 */

		args.push_back(value);
	}

	// Convert the arguments to a common type.
	// If any of the arguments is a float, also the others must be floats, in
	// order to preserve correctness. If a value is a boolean, it is first
	// extended to an integer in first place, and then, if needed, to a float.

	/*
	for (const auto& arg : args)
	{
		if (containsFloat)
			castedArgs.push_back(builder.create<CastOp>(arg.getLoc(), arg, floatType));
		else if (containsInteger)
			castedArgs.push_back(builder.create<CastOp>(arg.getLoc(), arg, integerType));
		else
			castedArgs.push_back(arg);
	}
	 */

	return args;
}
