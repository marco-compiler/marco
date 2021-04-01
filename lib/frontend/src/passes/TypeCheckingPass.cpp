#include <cstdio>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/TypeCheckingPass.h>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/utils/IRange.hpp>
#include <queue>
#include <stack>

using namespace modelica;
using namespace std;

llvm::Error resolveDummyReferences(Function& function);
llvm::Error resolveDummyReferences(Class& model);

llvm::Expected<Type> TypeChecker::typeFromSymbol(const Expression& exp)
{
	assert(exp.isA<ReferenceAccess>());
	ReferenceAccess acc = exp.get<ReferenceAccess>();

	// If the referenced variable is a dummy one (meaning that it is created
	// to store a result value that will never be used), its type is still
	// unknown and will be determined according to the assigned value.
	if (acc.isDummy())
		return Type::unknown();

	const auto& name = acc.getName();

	if (symbolTable.count(name) == 0)
	{
		if (name == "der")
			return Type::unknown();

		if (name == "size")
			return makeType<int>();

		if (name == "time")
			return makeType<float>();

		return llvm::make_error<NotImplemented>("Unknown variable name '" + name + "'");
	}

	auto symbol = symbolTable.lookup(name);

	if (symbol.isA<Function>())
		return symbol.get<Function>().getType();

	if (symbol.isA<Member>())
		return symbol.get<Member>().getType();

	if (symbol.isA<Induction>())
		return makeType<int>();

	return llvm::make_error<NotImplemented>("Unknown variable name '" + name + "'");
}

llvm::Error TypeChecker::run(ClassContainer& cls)
{
	SymbolTableScope varScope(symbolTable);

	return cls.visit([&](auto& obj) { return run(obj); });
}

llvm::Error TypeChecker::run(Function& function)
{
	// Populate the symbol table
	symbolTable.insert(function.getName(), Symbol(function));

	for (const auto& member : function.getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	llvm::SmallVector<Type, 3> types;

	// Check members

	for (auto& member : function.getMembers())
	{
		if (auto error = run(*member); error)
			return error;

		// From Function reference:
		// "Each input formal parameter of the function must be prefixed by the
		// keyword input, and each result formal parameter by the keyword output.
		// All public variables are formal parameters."

		if (member->isPublic() && !member->isInput() && !member->isOutput())
			return llvm::make_error<BadSemantic>(
					"Public members of functions must be input or output variables");

		// From Function reference:
		// "Input formal parameters are read-only after being bound to the actual
		// arguments or default values, i.e., they may not be assigned values in
		// the body of the function."

		if (member->isInput() && member->hasInitializer())
			return llvm::make_error<BadSemantic>(
					"Input variables can't receive a new value");

		// Add type
		if (member->isOutput())
			types.push_back(member->getType());
	}

	if (types.size() == 1)
		function.setType(move(types[0]));
	else
		function.setType(Type(PackedType(types)));

	auto& algorithms = function.getAlgorithms();

	// From Function reference:
	// "A function can have at most one algorithm section or one external
	// function interface (not both), which, if present, is the body of the
	// function."

	if (algorithms.size() > 1)
		return llvm::make_error<BadSemantic>(
				"Functions can have at most one algorithm section");

	// For now, functions can't have an external implementation and thus must
	// have exactly one algorithm section. When external implementations will
	// be allowed, the algorithms amount may also be zero.
	assert(algorithms.size() == 1);

	if (auto error = run(*algorithms[0]); error)
		return error;

	if (auto error = resolveDummyReferences(function); error)
		return error;

	for (const auto& statement : algorithms[0]->getStatements())
	{
		for (const auto& assignment : *statement)
		{
			for (const auto& exp : assignment.getDestinations())
			{
				// From Function reference:
				// "Input formal parameters are read-only after being bound to the
				// actual arguments or default values, i.e., they may not be assigned
				// values in the body of the function."
				auto* current = &exp;

				while (current->isA<Operation>())
				{
					auto& operation = current->get<Operation>();
					assert(operation.getKind() == OperationKind::subscription);
					current = &operation[0];
				}

				assert(current->isA<ReferenceAccess>());
				auto& ref = current->get<ReferenceAccess>();

				if (!ref.isDummy())
				{
					const auto& name = ref.getName();

					if (symbolTable.count(name) == 0)
						return llvm::make_error<NotImplemented>(
								"Unknown variable name '" + name + "'");

					const auto& member = symbolTable.lookup(name).get<Member>();

					if (member.isInput())
						return llvm::make_error<BadSemantic>(
								"Input variable '" + name + "' can't receive a new value");
				}
			}

			// From Function reference:
			// "A function cannot contain calls to the Modelica built-in operators
			// der, initial, terminal, sample, pre, edge, change, reinit, delay,
			// cardinality, inStream, actualStream, to the operators of the built-in
			// package Connections, and is not allowed to contain when-statements."

			stack<const Expression*> stack;
			stack.push(&assignment.getExpression());

			while (!stack.empty())
			{
				auto *expression = stack.top();
				stack.pop();

				if (expression->isA<ReferenceAccess>())
				{
					string name = expression->get<ReferenceAccess>().getName();

					if (name == "der" || name == "initial" || name == "terminal" ||
							name == "sample" || name == "pre" || name == "edge" ||
							name == "change" || name == "reinit" || name == "delay" ||
							name == "cardinality" || name == "inStream" ||
							name == "actualStream")
					{
						return llvm::make_error<BadSemantic>(
								"'" + name + "' is not allowed in procedural code");
					}

					// TODO: Connections built-in operators + when statement
				}
				else if (expression->isA<Operation>())
				{
					for (auto& arg : expression->get<Operation>())
						stack.push(&arg);
				}
				else if (expression->isA<Call>())
				{
					auto& call = expression->get<Call>();

					for (auto& arg : call)
						stack.push(&arg);

					stack.push(&call.getFunction());
				}
			}
		}
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Class& model)
{
	// Populate the symbol table
	symbolTable.insert(model.getName(), Symbol(model));

	for (auto& member : model.getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& m : model.getMembers())
		if (auto error = run(*m); error)
			return error;

	// Functions type checking must be done before the equations or algorithm
	// ones, because it establishes the result type of the functions that may
	// be invoked elsewhere.
	for (auto& cls : model.getInnerClasses())
		if (auto error = run(*cls); error)
			return error;

	for (auto& eq : model.getEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& eq : model.getForEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& algorithm : model.getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;

	if (auto error = resolveDummyReferences(model); error)
		return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Package& package)
{
	// Populate the symbol table
	symbolTable.insert(package.getName(), Symbol(package));

	for (auto& cls : package)
		cls.visit([&](auto& obj) { symbolTable.insert(obj.getName(), Symbol(obj)); });

	for (auto& cls : package)
		if (auto error = run(cls); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Record& record)
{
	// Populate the symbol table
	symbolTable.insert(record.getName(), Symbol(record));

	for (auto& member : record)
		symbolTable.insert(member.getName(), Symbol(member));

	for (auto& member : record)
		if (auto error = run(member); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Member& member)
{
	for (auto& dimension : member.getType().getDimensions())
		if (dimension.hasExpression())
			if (auto error = run<Expression>(dimension.getExpression()); error)
				return error;

	if (member.hasInitializer())
		if (auto error = run<Expression>(member.getInitializer()); error)
			return error;

	if (not member.hasStartOverload())
		return llvm::Error::success();

	if (auto error = run<Expression>(member.getStartOverload()); error)
		return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Algorithm& algorithm)
{
	for (auto& statement : algorithm.getStatements())
		if (auto error = run(*statement); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Statement& statement)
{
	return statement.visit([&](auto& statement) { return run(statement); });
}

llvm::Error TypeChecker::run(AssignmentStatement& statement)
{
	auto& destinations = statement.getDestinations();

	for (auto& destination : destinations)
	{
		if (auto error = run<Expression>(destination); error)
			return error;

		// The destinations must be l-values.
		// The check can't be enforced at parsing time because the grammar
		// specifies the destinations as expressions.

		if (!destination.isLValue())
			return llvm::make_error<BadSemantic>(
					"Destinations of statements must be l-values");
	}

	auto& expression = statement.getExpression();

	if (auto error = run<Expression>(expression); error)
		return error;

	if (destinations.size() > 1 && !expression.getType().isA<PackedType>())
		return llvm::make_error<IncompatibleType>(
				"The expression must return at least " +
				to_string(destinations.size()) + "values");

	// Assign type to dummy variables.
	// The assignment can't be done earlier because the expression type would
	// have not been evaluated yet.

	for (size_t i = 0, e = destinations.size(); i < e; ++i)
	{
		// If it's not a direct reference access, there's no way it can be a
		// dummy variable.
		if (!destinations[i].isA<ReferenceAccess>())
			continue;

		auto& ref = destinations[i].get<ReferenceAccess>();

		if (ref.isDummy())
		{
			auto& expressionType = expression.getType();
			assert(expressionType.isA<PackedType>());
			auto& userDefType = expressionType.get<PackedType>();
			assert(userDefType.size() >= i);
			destinations[i].setType(userDefType[i]);
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (expression.getType().isA<PackedType>())
	{
		auto& userDefType = expression.getType().get<PackedType>();
		size_t returns = userDefType.size();

		if (destinations.size() < returns)
		{
			vector<Expression> newDestinations;

			for (auto& destination : destinations)
				newDestinations.push_back(move(destination));

			for (size_t i = newDestinations.size(); i < returns; i++)
				newDestinations.emplace_back(userDefType[i], ReferenceAccess::dummy(SourcePosition::unknown()));

			statement.setDestination(Tuple(destinations.getLocation(), move(newDestinations)));
		}
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(IfStatement& statement)
{
	for (auto& block : statement)
	{
		if (auto error = run<Expression>(block.getCondition()); error)
			return error;

		for (auto& stmnt : block)
			if (auto error = run(*stmnt); error)
				return error;
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(ForStatement& statement)
{
	auto& induction = statement.getInduction();

	symbolTable.insert(induction.getName(), Symbol(induction));

	if (auto error = run<Expression>(induction.getBegin()); error)
		return error;

	if (auto error = run<Expression>(induction.getEnd()); error)
		return error;

	for (auto& stmnt : statement)
		if (auto error = run(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(WhileStatement& statement)
{
	if (auto error = run<Expression>(statement.getCondition()); error)
		return error;

	for (auto& stmnt : statement)
		if (auto error = run(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(WhenStatement& statement)
{
	if (auto error = run<Expression>(statement.getCondition()); error)
		return error;

	for (auto& stmnt : statement)
		if (auto error = run(*stmnt); error)
			return error;

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(BreakStatement& statement)
{
	return llvm::Error::success();
}

llvm::Error TypeChecker::run(ReturnStatement& statement)
{
	return llvm::Error::success();
}

llvm::Error TypeChecker::run(Equation& eq)
{
	auto& lhs = eq.getLeftHand();
	auto& rhs = eq.getRightHand();

	if (auto error = run<Expression>(lhs); error)
		return error;

	if (auto error = run<Expression>(rhs); error)
		return error;

	auto& lhsType = lhs.getType();
	auto& rhsType = rhs.getType();

	if (lhs.isA<Tuple>() && lhs.get<Tuple>().size() > 1)
		if (!rhsType.isA<PackedType>() ||
				lhs.get<Tuple>().size() > rhsType.get<PackedType>().size())
			return llvm::make_error<IncompatibleType>("Type dimension mismatch");

	if (lhs.isA<Tuple>())
	{
		auto& tuple = lhs.get<Tuple>();
		auto& types = rhsType.get<PackedType>();

		// Assign type to dummy variables.
		// The assignment can't be done earlier because the expression type would
		// have not been evaluated yet.

		for (size_t i = 0; i < tuple.size(); i++)
		{
			// If it's not a direct reference access, there's no way it can be a
			// dummy variable.
			if (!tuple[i].isA<ReferenceAccess>())
				continue;

			auto& ref = tuple[i].get<ReferenceAccess>();

			if (ref.isDummy())
			{
				assert(types.size() >= i);
				tuple[i].setType(types[i]);
			}
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (rhsType.isA<PackedType>())
	{
		auto& types = rhsType.get<PackedType>();
		size_t returns = types.size();

		auto location = lhs.getLocation();
		vector<Expression> newDestinations;

		if (lhs.isA<Tuple>())
			for (auto& destination : lhs.get<Tuple>())
				newDestinations.push_back(move(destination));
		else
			newDestinations.push_back(move(lhs));

		for (size_t i = newDestinations.size(); i < returns; i++)
			newDestinations.emplace_back(types[i], ReferenceAccess::dummy(SourcePosition::unknown()));

		Tuple tuple(location, move(newDestinations));
		eq.setLeftHand(Expression(rhsType, move(tuple)));
	}

	return llvm::Error::success();
}

llvm::Error TypeChecker::run(ForEquation& forEquation)
{
	SymbolTableScope varScope(symbolTable);

	for (auto& induction : forEquation.getInductions())
		symbolTable.insert(induction.getName(), Symbol(induction));

	if (auto error = run(forEquation.getEquation()); error)
		return error;

	for (auto& ind : forEquation.getInductions())
	{
		if (auto error = run<Expression>(ind.getBegin()); error)
			return error;

		if (auto error = run<Expression>(ind.getEnd()); error)
			return error;
	}

	return llvm::Error::success();
}

static llvm::Error subscriptionCheckType(Expression& exp)
{
	assert(exp.isA<Operation>());
	auto& op = exp.get<Operation>();
	assert(op.getKind() == OperationKind::subscription);

	size_t subscriptionIndicesCount = op.argumentsCount() - 1;

	if (subscriptionIndicesCount > op[0].getType().dimensionsCount())
		return llvm::make_error<IncompatibleType>("Array was subscripted too many times");

	for (size_t a = 1; a < op.argumentsCount(); a++)
		if (op[a].getType() != makeType<int>())
			return llvm::make_error<IncompatibleType>(
					"Parameter of array subscription was not int");

	exp.setType(op[0].getType().subscript(subscriptionIndicesCount));
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

static bool operator>=(Type x, Type y)
{
	assert(x.isA<BuiltInType>());
	assert(y.isA<BuiltInType>());

	if (x.get<BuiltInType>() == BuiltInType::Float)
		return true;

	if (x.get<BuiltInType>() == BuiltInType::Integer)
		return y.get<BuiltInType>() != BuiltInType::Float;

	if (x.get<BuiltInType>() == BuiltInType::Boolean)
		return y.get<BuiltInType>() == BuiltInType::Boolean;

	return false;
}

template<>
llvm::Error TypeChecker::run<Operation>(Expression& expression)
{
	assert(expression.isA<Operation>());
	auto& op = expression.get<Operation>();

	for (auto& arg : op)
		if (auto error = run<Expression>(arg); error)
			return error;

	switch (op.getKind())
	{
		case OperationKind::negate:
			expression.setType(op[0].getType());
			return llvm::Error::success();

		case OperationKind::add:
		case OperationKind::subtract:
		case OperationKind::multiply:
		case OperationKind::divide:
		{
			Type result = op[0].getType();

			for (auto& arg : op)
				if (arg.getType() >= result)
					result = arg.getType();

			expression.setType(result);
			return llvm::Error::success();
		}

		case OperationKind::powerOf:
			expression.setType(op[0].getType());
			return llvm::Error::success();

		case OperationKind::ifelse:
			if (op[0].getType() != makeType<bool>())
				return llvm::make_error<IncompatibleType>(
						"condition of if else was not boolean");
			if (op[1].getType() != op[2].getType())
				return llvm::make_error<IncompatibleType>(
						"ternary operator branches had different return type");

			expression.setType(op[1].getType());
			return llvm::Error::success();

		case OperationKind::greater:
		case OperationKind::greaterEqual:
		case OperationKind::equal:
		case OperationKind::different:
		case OperationKind::lessEqual:
		case OperationKind::less:
			expression.setType(makeType<bool>());
			return llvm::Error::success();

		case OperationKind::lor:
		case OperationKind::land:
			if (op[0].getType() != makeType<bool>())
				return llvm::make_error<IncompatibleType>(
						"boolean operator had non boolean argument");
			if (op[1].getType() != makeType<bool>())
				return llvm::make_error<IncompatibleType>(
						"boolean operator had non boolean argument");
			expression.setType(makeType<bool>());
			return llvm::Error::success();

		case OperationKind::subscription:
			return subscriptionCheckType(expression);

		case OperationKind::memberLookup:
			return llvm::make_error<NotImplemented>("member lookup is not implemented yet");
	}

	assert(false && "unreachable");
	return llvm::make_error<NotImplemented>("op was not any supported kind");
}

template<>
llvm::Error TypeChecker::run<Constant>(Expression& expression)
{
	assert(expression.isA<Constant>());
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<ReferenceAccess>(Expression& expression)
{
	assert(expression.isA<ReferenceAccess>());
	auto tp = typeFromSymbol(expression);

	if (!tp)
		return tp.takeError();

	expression.setType(move(*tp));
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Call>(Expression& expression)
{
	assert(expression.isA<Call>());
	auto& call = expression.get<Call>();

	for (size_t t : irange(call.argumentsCount()))
		if (auto error = run<Expression>(call[t]); error)
			return error;

	auto& function = call.getFunction();

	if (auto error = run<Expression>(function); error)
		return error;

	if (function.get<ReferenceAccess>().getName() == "der")
		function.setType(call[0].getType());

	expression.setType(function.getType());

	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Tuple>(Expression& expression)
{
	assert(expression.isA<Tuple>());
	auto& tuple = expression.get<Tuple>();

	llvm::SmallVector<Type, 3> types;

	for (auto& exp : tuple)
	{
		if (auto error = run<Expression>(exp); error)
			return error;

		types.push_back(exp.getType());
	}

	expression.setType(Type(PackedType(types)));
	return llvm::Error::success();
}

template<>
llvm::Error TypeChecker::run<Array>(Expression& expression)
{
	assert(expression.isA<Array>());
	auto& array = expression.get<Array>();

	Type type = makeType<bool>();
	llvm::SmallVector<long, 3> sizes;

	for (auto& value : array)
	{
		if (auto error = run<Expression>(value); error)
			return error;

		if (value.getType() >= type)
			type = value.getType();

		auto& valueType = value.getType();
		unsigned int rank = valueType.dimensionsCount();

		if (!valueType.isScalar())
		{
			if (sizes.empty())
			{
				for (size_t i = 0; i < rank; ++i)
				{
					assert(!valueType[i].hasExpression());
					sizes.push_back(valueType[i].getNumericSize());
				}
			}
			else
			{
				assert(sizes.size() == rank);
			}
		}
	}

	llvm::SmallVector<ArrayDimension, 3> dimensions;
	dimensions.emplace_back(array.size());

	for (auto size : sizes)
		dimensions.emplace_back(size);

	type.setDimensions(dimensions);
	expression.setType(type);
	return llvm::Error::success();
}

template<class T>
string getTemporaryVariableName(T& cls)
{
	const auto& members = cls.getMembers();
	int counter = 0;

	while (*(members.end()) !=
				 *find_if(members.begin(), members.end(), [=](std::shared_ptr<Member> obj) {
					 return obj->getName() == "_temp" + to_string(counter);
				 }))
		counter++;

	return "_temp" + to_string(counter);
}

llvm::Error resolveDummyReferences(Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
	{
		for (auto& statement : algorithm->getStatements())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : assignment.getDestinations())
				{
					if (!destination.isA<ReferenceAccess>())
						continue;

					auto& ref = destination.get<ReferenceAccess>();

					if (!ref.isDummy())
						continue;

					string name = getTemporaryVariableName(function);
					Member temp(destination.getLocation(), name, destination.getType(), TypePrefix::none());
					ref.setName(temp.getName());
					function.addMember(temp);

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

llvm::Error resolveDummyReferences(Class& model)
{
	for (auto& equation : model.getEquations())
	{
		auto& lhs = equation->getLeftHand();

		if (lhs.isA<Tuple>())
		{
			for (auto& expression : lhs.get<Tuple>())
			{
				if (!expression.isA<ReferenceAccess>())
					continue;

				auto& ref = expression.get<ReferenceAccess>();

				if (!ref.isDummy())
					continue;

				string name = getTemporaryVariableName(model);
				Member temp(expression.getLocation(), name, expression.getType(), TypePrefix::none());
				ref.setName(temp.getName());
				model.addMember(temp);
			}
		}
	}

	// TODO: check of ForEquation

	for (auto& algorithm : model.getAlgorithms())
	{
		for (auto& statement : algorithm->getStatements())
		{
			for (auto& assignment : *statement)
			{
				for (auto& destination : assignment.getDestinations())
				{
					if (!destination.isA<ReferenceAccess>())
						continue;

					auto& ref = destination.get<ReferenceAccess>();

					if (!ref.isDummy())
						continue;

					string name = getTemporaryVariableName(model);
					Member temp(destination.getLocation(), name, destination.getType(), TypePrefix::none());
					ref.setName(temp.getName());
					model.addMember(temp);

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return llvm::Error::success();
}

std::unique_ptr<Pass> modelica::createTypeCheckingPass()
{
	return std::make_unique<TypeChecker>();
}
