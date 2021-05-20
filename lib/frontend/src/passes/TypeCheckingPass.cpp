#include <cstdio>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/Errors.h>
#include <modelica/frontend/passes/TypeCheckingPass.h>
#include <queue>
#include <stack>

using namespace modelica;
using namespace modelica::frontend;

llvm::Error resolveDummyReferences(StandardFunction& function);
llvm::Error resolveDummyReferences(Model& model);

static bool operator>=(Type x, Type y)
{
	assert(x.isA<BuiltInType>());
	assert(y.isA<BuiltInType>());

	if (y.get<BuiltInType>() == BuiltInType::Unknown)
		return true;

	if (x.get<BuiltInType>() == BuiltInType::Float)
		return true;

	if (x.get<BuiltInType>() == BuiltInType::Integer)
		return y.get<BuiltInType>() != BuiltInType::Float;

	if (x.get<BuiltInType>() == BuiltInType::Boolean)
		return y.get<BuiltInType>() == BuiltInType::Boolean;

	return false;
}

template<>
llvm::Error TypeChecker::run<Class>(Class& cls)
{
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error TypeChecker::run(Class& cls)
{
	return run<Class>(cls);
}

template<>
llvm::Error TypeChecker::run<DerFunction>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<StandardFunction>(Class& cls)
{
	SymbolTableScope varScope(symbolTable);
	auto* function = cls.get<StandardFunction>();

	// Populate the symbol table
	symbolTable.insert(function->getName(), Symbol(*function));

	for (const auto& member : function->getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	llvm::SmallVector<Type, 3> types;

	// Check members

	for (auto& member : function->getMembers())
	{
		if (auto error = run(*member); error)
			return error;

		// From Function reference:
		// "Each input formal parameter of the function must be prefixed by the
		// keyword input, and each result formal parameter by the keyword output.
		// All public variables are formal parameters."

		if (member->isPublic() && !member->isInput() && !member->isOutput())
			return llvm::make_error<BadSemantic>(
					member->getLocation(),
					"public members of functions must be input or output variables");

		// From Function reference:
		// "Input formal parameters are read-only after being bound to the actual
		// arguments or default values, i.e., they may not be assigned values in
		// the body of the function."

		if (member->isInput() && member->hasInitializer())
			return llvm::make_error<BadSemantic>(
					member->getLocation(),
					"input variables can't receive a new value");

		// Add type
		if (member->isOutput())
			types.push_back(member->getType());
	}

	if (types.size() == 1)
		function->setType(std::move(types[0]));
	else
		function->setType(Type(PackedType(types)));

	auto algorithms = function->getAlgorithms();

	// From Function reference:
	// "A function can have at most one algorithm section or one external
	// function interface (not both), which, if present, is the body of the
	// function."

	if (algorithms.size() > 1)
		return llvm::make_error<BadSemantic>(
				function->getLocation(),
				"functions can have at most one algorithm section");

	// For now, functions can't have an external implementation and thus must
	// have exactly one algorithm section. When external implementations will
	// be allowed, the algorithms amount may also be zero.
	assert(algorithms.size() == 1);

	if (auto error = run(*algorithms[0]); error)
		return error;

	if (auto error = resolveDummyReferences(*function); error)
		return error;

	for (const auto& statement : *algorithms[0])
	{
		for (const auto& assignment : *statement)
		{
			for (const auto& exp : *assignment.getDestinations()->get<Tuple>())
			{
				// From Function reference:
				// "Input formal parameters are read-only after being bound to the
				// actual arguments or default values, i.e., they may not be assigned
				// values in the body of the function."
				const auto* current = exp.get();

				while (current->isa<Operation>())
				{
					const auto* operation = current->get<Operation>();
					assert(operation->getOperationKind() == OperationKind::subscription);
					current = operation->getArg(0);
				}

				assert(current->isa<ReferenceAccess>());
				const auto* ref = current->get<ReferenceAccess>();

				if (!ref->isDummy())
				{
					const auto& name = ref->getName();

					if (symbolTable.count(name) == 0)
						return llvm::make_error<NotImplemented>(
								"unknown variable name '" + name.str() + "'");

					const auto& member = symbolTable.lookup(name).get<Member>();

					if (member->isInput())
						return llvm::make_error<BadSemantic>(
								assignment.getLocation(),
								"input variable '" + name.str() + "' can't receive a new value");
				}
			}

			// From Function reference:
			// "A function cannot contain calls to the Modelica built-in operators
			// der, initial, terminal, sample, pre, edge, change, reinit, delay,
			// cardinality, inStream, actualStream, to the operators of the built-in
			// package Connections, and is not allowed to contain when-statements."

			std::stack<const Expression*> stack;
			stack.push(assignment.getExpression());

			while (!stack.empty())
			{
				const auto *expression = stack.top();
				stack.pop();

				if (expression->isa<ReferenceAccess>())
				{
					llvm::StringRef name = expression->get<ReferenceAccess>()->getName();

					if (name == "der" || name == "initial" || name == "terminal" ||
							name == "sample" || name == "pre" || name == "edge" ||
							name == "change" || name == "reinit" || name == "delay" ||
							name == "cardinality" || name == "inStream" ||
							name == "actualStream")
					{
						return llvm::make_error<BadSemantic>(
								expression->getLocation(),
								"'" + name.str() + "' is not allowed in procedural code");
					}

					// TODO: Connections built-in operators + when statement
				}
				else if (expression->isa<Operation>())
				{
					for (auto& arg : *expression->get<Operation>())
						stack.push(arg.get());
				}
				else if (expression->isa<Call>())
				{
					auto* call = expression->get<Call>();

					for (auto& arg : *call)
						stack.push(arg.get());

					stack.push(call->getFunction());
				}
			}
		}
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Model>(Class& cls)
{
	SymbolTableScope varScope(symbolTable);
	auto* model = cls.get<Model>();

	// Populate the symbol table
	symbolTable.insert(model->getName(), Symbol(*model));

	for (auto& member : model->getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& m : model->getMembers())
		if (auto error = run(*m); error)
			return error;

	// Functions type checking must be done before the equations or algorithm
	// ones, because it establishes the result type of the functions that may
	// be invoked elsewhere.
	for (auto& innerClass : model->getInnerClasses())
		if (auto error = run(*innerClass); error)
			return error;

	for (auto& eq : model->getEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& eq : model->getForEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& algorithm : model->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	if (auto error = resolveDummyReferences(*model); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Package>(Class& cls)
{
	SymbolTableScope varScope(symbolTable);
	auto* package = cls.get<Package>();

	// Populate the symbol table
	symbolTable.insert(package->getName(), Symbol(*package));

	for (auto& cls : *package)
		cls->visit([&](auto& obj) { symbolTable.insert(obj.getName(), Symbol(obj)); });

	for (auto& cls : *package)
		if (auto error = run(*cls); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Record>(Class& cls)
{
	SymbolTableScope varScope(symbolTable);
	auto* record = cls.get<Record>();

	// Populate the symbol table
	symbolTable.insert(record->getName(), Symbol(*record));

	for (auto& member : *record)
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& member : *record)
		if (auto error = run(*member); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Equation& equation)
{
	if (auto error = run<Expression>(*equation.getLhsExpression()); error)
		return error;

	if (auto error = run<Expression>(*equation.getRhsExpression()); error)
		return error;

	auto* lhs = equation.getLhsExpression();
	auto* rhs = equation.getRhsExpression();

	const auto& rhsType = rhs->getType();

	if (auto* lhsTuple = lhs->dyn_get<Tuple>())
	{
		if (!rhsType.isA<PackedType>() ||
		    lhsTuple->size() != rhsType.get<PackedType>().size())
			return llvm::make_error<IncompatibleType>("type dimension mismatch");
	}

	if (auto* lhsTuple = lhs->dyn_get<Tuple>())
	{
		assert(rhs->getType().isA<PackedType>());
		auto& rhsTypes = rhs->getType().get<PackedType>();

		// Assign type to dummy variables.
		// The assignment can't be done earlier because the expression type would
		// have not been evaluated yet.

		for (size_t i = 0; i < lhsTuple->size(); ++i)
		{
			// If it's not a direct reference access, there's no way it can be a
			// dummy variable.

			if (!lhsTuple->getArg(i)->isa<ReferenceAccess>())
				continue;

			auto* ref = lhsTuple->getArg(i)->get<ReferenceAccess>();

			if (ref->isDummy())
			{
				assert(rhsTypes.size() >= i);
				lhsTuple->getArg(i)->setType(rhsTypes[i]);
			}
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (rhsType.isA<PackedType>())
	{
		const auto& rhsPackedType = rhsType.get<PackedType>();
		size_t returns = rhsPackedType.size();

		llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
		llvm::SmallVector<Type, 3> destinationsTypes;

		if (auto* lhsTuple = lhs->dyn_get<Tuple>())
		{
			for (auto& destination : *lhsTuple)
			{
				destinationsTypes.push_back(destination->getType());
				newDestinations.push_back(std::move(destination));
			}
		}
		else
		{
			destinationsTypes.push_back(lhs->getType());
			newDestinations.push_back(lhs->clone());
		}

		for (size_t i = newDestinations.size(); i < returns; ++i)
		{
			destinationsTypes.push_back(rhsPackedType[i]);
			newDestinations.push_back(ReferenceAccess::dummy(equation.getLocation(), rhsPackedType[i]));
		}

		equation.setLhsExpression(
				Expression::tuple(lhs->getLocation(), Type(PackedType(destinationsTypes)), newDestinations));
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(ForEquation& forEquation)
{
	SymbolTableScope varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind->getName(), Symbol(*ind));

		if (auto error = run<Expression>(*ind->getBegin()); error)
			return error;

		if (auto error = run<Expression>(*ind->getEnd()); error)
			return error;
	}

	if (auto error = run(*forEquation.getEquation()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(expression);
	});
}

template<>
llvm::Error TypeChecker::run<Array>(Expression& expression)
{
	auto* array = expression.get<Array>();

	llvm::SmallVector<long, 3> sizes;

	auto resultType = makeType<bool>();

	for (auto& element : *array)
	{
		if (auto error = run<Expression>(*element); error)
			return error;

		auto& elementType = element->getType();
		assert(elementType.isA<BuiltInType>());
		auto& builtInElementType = elementType.get<BuiltInType>();

		assert(builtInElementType == BuiltInType::Boolean ||
					 builtInElementType == BuiltInType::Integer ||
					 builtInElementType == BuiltInType::Float);

		if (elementType >= resultType)
			resultType = elementType;

		unsigned int rank = elementType.dimensionsCount();

		if (!elementType.isScalar())
		{
			if (sizes.empty())
			{
				for (size_t i = 0; i < rank; ++i)
				{
					assert(!elementType[i].hasExpression());
					sizes.push_back(elementType[i].getNumericSize());
				}
			}
			else
			{
				assert(sizes.size() == rank);
			}
		}
	}

	llvm::SmallVector<ArrayDimension, 3> dimensions;
	dimensions.emplace_back(array->size());

	for (auto size : sizes)
		dimensions.emplace_back(size);

	resultType.setDimensions(dimensions);
	expression.setType(resultType);

	return llvm::Error::success();
}

static llvm::Optional<Type> builtInFunctionType(Call& call)
{
	auto name = call.getFunction()->get<ReferenceAccess>()->getName();
	const auto args = call.getArgs();

	if (name == "der")
		return args[0]->getType().to(BuiltInType::Float);

	return llvm::None;
}

template<>
llvm::Error TypeChecker::run<Call>(Expression& expression)
{
	auto* call = expression.get<Call>();

	for (auto& arg : *call)
		if (auto error = run<Expression>(*arg); error)
			return error;

	auto* function = call->getFunction();

	if (auto type = builtInFunctionType(*call);
			type.hasValue() &&
			symbolTable.count(function->get<ReferenceAccess>()->getName()) == 0)
	{
		expression.setType(type.getValue());
	}
	else
	{
		if (auto error = run<Expression>(*function); error)
			return error;

		expression.setType(function->getType());
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Constant>(Expression& expression)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Operation>(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	auto checkOperation = [&](Expression& expression, std::function<llvm::Error(TypeChecker&, Expression&)> checker) -> llvm::Error {
		if (auto error = checker(*this, expression); error)
			return error;

		return llvm::Error::success();
	};

	switch (operation->getOperationKind())
	{
		case OperationKind::add:
			return checkOperation(expression, &TypeChecker::checkAddOp);

		case OperationKind::different:
			return checkOperation(expression, &TypeChecker::checkDifferentOp);

		case OperationKind::divide:
			return checkOperation(expression, &TypeChecker::checkDivOp);

		case OperationKind::equal:
			return checkOperation(expression, &TypeChecker::checkEqualOp);

		case OperationKind::greater:
			return checkOperation(expression, &TypeChecker::checkGreaterOp);

		case OperationKind::greaterEqual:
			return checkOperation(expression, &TypeChecker::checkGreaterEqualOp);

		case OperationKind::ifelse:
			return checkOperation(expression, &TypeChecker::checkIfElseOp);

		case OperationKind::less:
			return checkOperation(expression, &TypeChecker::checkLessOp);

		case OperationKind::lessEqual:
			return checkOperation(expression, &TypeChecker::checkLessEqualOp);

		case OperationKind::land:
			return checkOperation(expression, &TypeChecker::checkLogicalAndOp);

		case OperationKind::lor:
			return checkOperation(expression, &TypeChecker::checkLogicalOrOp);

		case OperationKind::memberLookup:
			return checkOperation(expression, &TypeChecker::checkMemberLookupOp);

		case OperationKind::multiply:
			return checkOperation(expression, &TypeChecker::checkMulOp);

		case OperationKind::negate:
			return checkOperation(expression, &TypeChecker::checkNegateOp);

		case OperationKind::powerOf:
			return checkOperation(expression, &TypeChecker::checkPowerOfOp);

		case OperationKind::subscription:
			return checkOperation(expression, &TypeChecker::checkSubscriptionOp);

		case OperationKind::subtract:
			return checkOperation(expression, &TypeChecker::checkSubOp);
	}

	return llvm::Error::success();
}

static llvm::Optional<Type> builtInReferenceType(ReferenceAccess& reference)
{
	assert(!reference.isDummy());
	auto name = reference.getName();

	if (name == "time")
		return makeType<float>();

	return llvm::None;
}

template<>
llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression)
{
	auto* reference = expression.get<ReferenceAccess>();

	// If the referenced variable is a dummy one (meaning that it is created
	// to store a result value that will never be used), its type is still
	// unknown and will be determined according to the assigned value.

	if (reference->isDummy())
		return llvm::Error::success();

	auto name = reference->getName();

	if (symbolTable.count(name) == 0)
	{
		if (auto type = builtInReferenceType(*reference); type.hasValue())
		{
			expression.setType(type.getValue());
			return llvm::Error::success();
		}

		return llvm::make_error<NotImplemented>("Unknown variable name '" + name.str() + "'");
	}

	auto symbol = symbolTable.lookup(name);

	auto symbolType = [](Symbol& symbol) -> Type {
		if (symbol.isa<StandardFunction>())
			return symbol.get<StandardFunction>()->getType();

		if (symbol.isa<Member>())
			return symbol.get<Member>()->getType();

		if (symbol.isa<Induction>())
			return makeType<int>();

		return Type::unknown();
	};

	expression.setType(symbolType(symbol));
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Tuple>(Expression& expression)
{
	auto* tuple = expression.get<Tuple>();
	llvm::SmallVector<Type, 3> types;

	for (auto& exp : *tuple)
	{
		if (auto error = run<Expression>(*exp); error)
			return error;

		types.push_back(exp->getType());
	}

	expression.setType(Type(PackedType(types)));
	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Member& member)
{
	for (auto& dimension : member.getType().getDimensions())
		if (dimension.hasExpression())
			if (auto error = run<Expression>(*dimension.getExpression()); error)
				return error;

	if (member.hasInitializer())
		if (auto error = run<Expression>(*member.getInitializer()); error)
			return error;

	if (member.hasStartOverload())
		if (auto error = run<Expression>(*member.getStartOverload()); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
llvm::Error TypeChecker::run<AssignmentStatement>(Statement& statement)
{
	auto* assignmentStatement = statement.get<AssignmentStatement>();

	auto* destinations = assignmentStatement->getDestinations();
	auto* destinationsTuple = destinations->get<Tuple>();

	for (auto& destination : *destinationsTuple)
	{
		if (auto error = run<Expression>(*destination); error)
			return error;

		// The destinations must be l-values.
		// The check can't be enforced at parsing time because the grammar
		// specifies the destinations as expressions.

		if (!destination->isLValue())
			return llvm::make_error<BadSemantic>(
					destination->getLocation(),
					"Destinations of statements must be l-values");
	}

	auto* expression = assignmentStatement->getExpression();

	if (auto error = run<Expression>(*expression); error)
		return error;

	if (destinationsTuple->size() > 1 && !expression->getType().isA<PackedType>())
		return llvm::make_error<IncompatibleType>(
				"The expression must return at least " +
				std::to_string(destinationsTuple->size()) + "values");

	// Assign type to dummy variables.
	// The assignment can't be done earlier because the expression type would
	// have not been evaluated yet.

	for (size_t i = 0, e = destinationsTuple->size(); i < e; ++i)
	{
		// If it's not a direct reference access, there's no way it can be a
		// dummy variable.
		if (!destinationsTuple->getArg(i)->isa<ReferenceAccess>())
			continue;

		auto* reference = destinationsTuple->getArg(i)->get<ReferenceAccess>();

		if (reference->isDummy())
		{
			auto& expressionType = expression->getType();
			assert(expressionType.isA<PackedType>());
			auto& packedType = expressionType.get<PackedType>();
			assert(packedType.size() >= i);
			destinationsTuple->getArg(i)->setType(packedType[i]);
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (expression->getType().isA<PackedType>())
	{
		auto& packedType = expression->getType().get<PackedType>();
		size_t returns = packedType.size();

		if (destinationsTuple->size() < returns)
		{
			llvm::SmallVector<std::unique_ptr<Expression>, 3> newDestinations;
			llvm::SmallVector<Type, 3> destinationsTypes;

			for (auto& destination : *destinationsTuple)
			{
				destinationsTypes.push_back(destination->getType());
				newDestinations.push_back(std::move(destination));
			}

			for (size_t i = newDestinations.size(); i < returns; ++i)
			{
				destinationsTypes.push_back(packedType[i]);
				newDestinations.emplace_back(ReferenceAccess::dummy(statement.getLocation(), packedType[i]));
			}

			assignmentStatement->setDestinations(
					Expression::tuple(destinations->getLocation(), Type(PackedType(destinationsTypes)), std::move(newDestinations)));
		}
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<BreakStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<ForStatement>(Statement& statement)
{
	auto* forStatement = statement.get<ForStatement>();

	if (auto error = run(*forStatement->getInduction()); error)
		return error;

	auto* induction = forStatement->getInduction();
	symbolTable.insert(induction->getName(), Symbol(*induction));

	for (auto& stmnt : forStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Induction& induction)
{
	if (auto error = run<Expression>(*induction.getBegin()); error)
		return error;

	if (auto error = run<Expression>(*induction.getEnd()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<IfStatement>(Statement& statement)
{
	auto* ifStatement = statement.get<IfStatement>();

	for (auto& block : *ifStatement)
	{
		if (auto error = run<Expression>(*block.getCondition()); error)
			return error;

		for (auto& stmnt : block)
			if (auto error = run<Statement>(*stmnt); error)
				return error;
	}

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<ReturnStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<WhenStatement>(Statement& statement)
{
	auto* whenStatement = statement.get<WhenStatement>();

	if (auto error = run<Expression>(*whenStatement->getCondition()); error)
		return error;

	for (auto& stmnt : whenStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<WhileStatement>(Statement& statement)
{
	auto* whileStatement = statement.get<WhileStatement>();

	if (auto error = run<Expression>(*whileStatement->getCondition()); error)
		return error;

	for (auto& stmnt : whileStatement->getBody())
		if (auto error = run<Statement>(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Algorithm& algorithm)
{
	for (auto& statement : algorithm)
		if (auto error = run<Statement>(*statement); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGenericOperation(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	Type type = expression.getType();

	for (auto& arg : operation->getArguments())
		if (auto& argType = arg->getType(); argType >= type)
			type = argType;

	expression.setType(std::move(type));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkAddOp(Expression& expression)
{
	return checkGenericOperation(expression);
}

llvm::Error TypeChecker::checkDifferentOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::different);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkDivOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::divide);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	Type type = operation->getArg(0)->getType().to(BuiltInType::Float);
	expression.setType(std::move(type));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::equal);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGreaterOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greater);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkGreaterEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::greaterEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkIfElseOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::ifelse);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	if (operation->getArg(0)->getType() != makeType<bool>())
		return llvm::make_error<IncompatibleType>(
				"condition of if else was not boolean");

	if (operation->getArg(1)->getType() != operation->getArg(2)->getType())
		return llvm::make_error<IncompatibleType>(
				"ternary operator branches had different return type");

	expression.setType(operation->getArg(1)->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLessOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::less);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLessEqualOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lessEqual);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLogicalAndOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::land);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkLogicalOrOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::lor);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(makeType<bool>());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkMemberLookupOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::memberLookup);
	return llvm::make_error<NotImplemented>("member lookup is not implemented yet");
}

llvm::Error TypeChecker::checkMulOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::multiply);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	Type type = expression.getType();

	for (auto& arg : operation->getArguments())
		if (auto& argType = arg->getType(); argType >= type)
			type = argType;

	expression.setType(operation->getArg(0)->getType().to(type.get<BuiltInType>()));
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkNegateOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::negate);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(operation->getArg(0)->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkPowerOfOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::powerOf);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	expression.setType(operation->getArg(0)->getType());
	return llvm::Error::success();
}

llvm::Error TypeChecker::checkSubOp(Expression& expression)
{
	return checkGenericOperation(expression);
}

llvm::Error TypeChecker::checkSubscriptionOp(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	assert(operation->getOperationKind() == OperationKind::subscription);

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	size_t subscriptionIndicesCount = operation->argumentsCount() - 1;

	if (subscriptionIndicesCount > operation->getArg(0)->getType().dimensionsCount())
		return llvm::make_error<IncompatibleType>("Array was subscripted too many times");

	for (size_t a = 1; a < operation->argumentsCount(); a++)
		if (operation->getArg(a)->getType() != makeType<int>())
			return llvm::make_error<IncompatibleType>(
					"Parameter of array subscription was not int");

	expression.setType(operation->getArg(0)->getType().subscript(subscriptionIndicesCount));
	return llvm::Error::success();
}


































































template<class T>
std::string getTemporaryVariableName(T& cls)
{
	const auto& members = cls.getMembers();
	int counter = 0;

	while (*(members.end()) !=
				 *find_if(members.begin(), members.end(), [=](const auto& obj) {
					 return obj->getName() == "_temp" + std::to_string(counter);
				 }))
		counter++;

	return "_temp" + std::to_string(counter);
}

llvm::Error resolveDummyReferences(StandardFunction& function)
{
	for (auto& algorithm : function.getAlgorithms())
	{
		for (auto& statement : algorithm->getBody())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : *assignment.getDestinations()->get<Tuple>())
				{
					if (!destination->isa<ReferenceAccess>())
						continue;

					auto* ref = destination->get<ReferenceAccess>();

					if (!ref->isDummy())
						continue;

					std::string name = getTemporaryVariableName(function);
					auto temp = Member::build(destination->getLocation(), name, destination->getType(), TypePrefix::none(), llvm::None);
					ref->setName(temp->getName());
					function.addMember(std::move(temp));

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

llvm::Error resolveDummyReferences(Model& model)
{
	for (auto& equation : model.getEquations())
	{
		auto* lhs = equation->getLhsExpression();

		if (auto* lhsTuple = lhs->dyn_get<Tuple>())
		{
			for (auto& expression : *lhsTuple)
			{
				if (!expression->isa<ReferenceAccess>())
					continue;

				auto* ref = expression->get<ReferenceAccess>();

				if (!ref->isDummy())
					continue;

				std::string name = getTemporaryVariableName(model);
				auto temp = Member::build(expression->getLocation(), name, expression->getType(), TypePrefix::none(), llvm::None);
				ref->setName(temp->getName());
				model.addMember(std::move(temp));
			}
		}
	}

	// TODO: check of ForEquation

	for (auto& algorithm : model.getAlgorithms())
	{
		for (auto& statement : algorithm->getBody())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : *assignment.getDestinations()->get<Tuple>())
				{
					if (!destination->isa<ReferenceAccess>())
						continue;

					auto* ref = destination->get<ReferenceAccess>();

					if (!ref->isDummy())
						continue;

					std::string name = getTemporaryVariableName(model);
					auto temp = Member::build(destination->getLocation(), name, destination->getType(), TypePrefix::none(), llvm::None);
					ref->setName(temp->getName());
					model.addMember(std::move(temp));

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

std::unique_ptr<Pass> modelica::frontend::createTypeCheckingPass()
{
	return std::make_unique<TypeChecker>();
}
