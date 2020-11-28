#include <cstdio>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/ForEquation.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/frontend/TypeChecker.hpp>
#include <modelica/utils/IRange.hpp>
#include <queue>
#include <stack>

using namespace llvm;
using namespace modelica;
using namespace std;

llvm::Error resolveDummyReferences(Function& function);
llvm::Error resolveDummyReferences(Class& model);

static Expected<Type> typeFromSymbol(
		const Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<ReferenceAccess>());
	ReferenceAccess acc = exp.get<ReferenceAccess>();

	// If the referenced variable is a dummy one (meaning that it is created
	// to store a result value that will never be used), its type is still
	// unknown and will be determined according to the assigned value.
	if (acc.isDummy())
		return Type::unknown();

	const auto& name = acc.getName();

	if (name == "der")
		return Type::unknown();

	if (name == "time")
		return Type::Float();

	if (!table.hasSymbol(name))
		return make_error<NotImplemented>("Unknown variable name '" + name + "'");

	auto symbol = table[name];

	if (symbol.isA<Function>())
		return symbol.get<Function>().getType();

	if (symbol.isA<Member>())
		return symbol.get<Member>().getType();

	if (symbol.isA<Induction>())
		return makeType<int>();

	return make_error<NotImplemented>("Unknown variable name '" + name + "'");
}

Error TypeChecker::checkType(Algorithm& algorithm, const SymbolTable& table)
{
	for (auto& statement : algorithm.getStatements())
		if (auto error = checkType(statement, table); error)
			return error;

	return Error::success();
}

Error TypeChecker::checkType(ClassContainer& cls, const SymbolTable& table)
{
	return cls.visit([&](auto& obj) { return checkType(obj, table); });
}

Error TypeChecker::checkType(Function& function, const SymbolTable& table)
{
	SymbolTable t(function, &table);
	vector<Type> types;

	// Check members

	for (auto& member : function.getMembers())
	{
		if (auto error = checkType(member, t); error)
			return error;

		// From Function reference:
		// "Each input formal parameter of the function must be prefixed by the
		// keyword input, and each result formal parameter by the keyword output.
		// All public variables are formal parameters."

		if (member.isPublic() && !member.isInput() && !member.isOutput())
			return make_error<BadSemantic>(
					"Public members of functions must be input or output variables");

		// From Function reference:
		// "Input formal parameters are read-only after being bound to the actual
		// arguments or default values, i.e., they may not be assigned values in
		// the body of the function."

		if (member.isInput() && member.hasInitializer())
			return make_error<BadSemantic>(
					"Input variables can't receive a new value");

		// Add type
		if (member.isOutput())
			types.push_back(member.getType());
	}

	if (types.size() == 1)
		function.setType(move(types[0]));
	else
		function.setType(Type(types));

	auto& algorithms = function.getAlgorithms();

	// From Function reference:
	// "A function can have at most one algorithm section or one external
	// function interface (not both), which, if present, is the body of the
	// function."

	if (algorithms.size() > 1)
		return make_error<BadSemantic>(
				"Functions can have at most one algorithm section");

	// For now, functions can't have an external implementation and thus must
	// have exactly one algorithm section. When external implementations will
	// be allowed, the algorithms amount may also be zero.
	assert(algorithms.size() == 1);

	if (auto error = checkType(algorithms[0], t); error)
		return error;

	if (auto error = resolveDummyReferences(function); error)
		return error;

	for (auto& statement : algorithms[0].getStatements())
	{
		for (auto& assignment : statement)
		{
			for (auto& destination : assignment.getDestinations())
			{
				// From Function reference:
				// "Input formal parameters are read-only after being bound to the
				// actual arguments or default values, i.e., they may not be assigned
				// values in the body of the function."

				auto& exp = *destination;

				while (exp.isA<Operation>())
				{
					auto& operation = exp.get<Operation>();
					assert(operation.getKind() == OperationKind::subscription);
					exp = operation[0];
				}

				assert(exp.isA<ReferenceAccess>());
				auto& ref = exp.get<ReferenceAccess>();

				if (!ref.isDummy())
				{
					const auto& name = ref.getName();

					if (!t.hasSymbol(name))
						return make_error<NotImplemented>(
								"Unknown variable name '" + name + "'");

					const auto& member = t[name].get<Member>();

					if (member.isInput())
						return make_error<BadSemantic>(
								"Input variable '" + name + "' can't receive a new value");
				}
			}

			// From Function reference:
			// "A function cannot contain calls to the Modelica built-in operators
			// der, initial, terminal, sample, pre, edge, change, reinit, delay,
			// cardinality, inStream, actualStream, to the operators of the built-in
			// package Connections, and is not allowed to contain when-statements."

			stack<Expression> stack;
			stack.push(assignment.getExpression());

			while (!stack.empty())
			{
				auto expression = stack.top();
				stack.pop();

				if (expression.isA<ReferenceAccess>())
				{
					string name = expression.get<ReferenceAccess>().getName();

					if (name == "der" || name == "initial" || name == "terminal" ||
							name == "sample" || name == "pre" || name == "edge" ||
							name == "change" || name == "reinit" || name == "delay" ||
							name == "cardinality" || name == "inStream" ||
							name == "actualStream")
					{
						return make_error<BadSemantic>(
								"'" + name + "' is not allowed in procedural code");
					}

					// TODO: Connections built-in operators + when statement
				}
				else if (expression.isA<Operation>())
				{
					for (auto& arg : expression.get<Operation>())
						stack.push(arg);
				}
				else if (expression.isA<Call>())
				{
					auto& call = expression.get<Call>();

					for (auto& arg : call)
						stack.push(*arg);

					stack.push(call.getFunction());
				}
			}
		}
	}

	return Error::success();
}

Error TypeChecker::checkType(Class& model, const SymbolTable& table)
{
	SymbolTable t(model, &table);

	for (auto& m : model.getMembers())
		if (auto error = checkType(m, t); error)
			return error;

	// Functions type checking must be done before the equations or algorithm
	// ones, because it establishes the result type of the functions that may
	// be invoked elsewhere.
	for (auto& cls : model.getInnerClasses())
		if (auto error = checkType(*cls, t); error)
			return error;

	for (auto& eq : model.getEquations())
		if (auto error = checkType(eq, t); error)
			return error;

	for (auto& eq : model.getForEquations())
		if (auto error = checkType(eq, t); error)
			return error;

	for (auto& algorithm : model.getAlgorithms())
		if (auto error = checkType(algorithm, t); error)
			return error;

	if (auto error = resolveDummyReferences(model); error)
		return error;

	return Error::success();
}

Error TypeChecker::checkType(Member& mem, const SymbolTable& table)
{
	if (mem.hasInitializer())
		if (auto error = checkType<Expression>(mem.getInitializer(), table); error)
			return error;

	if (not mem.hasStartOverload())
		return Error::success();

	if (auto error = checkType<Expression>(mem.getStartOverload(), table); error)
		return error;

	return Error::success();
}

Error TypeChecker::checkType(ForEquation& eq, const SymbolTable& table)
{
	SymbolTable t(&table);

	for (auto& ind : eq.getInductions())
		t.addSymbol(ind);

	if (auto error = checkType(eq.getEquation(), t); error)
		return error;

	for (auto& ind : eq.getInductions())
	{
		if (auto error = checkType<Expression>(ind.getBegin(), table); error)
			return error;

		if (auto error = checkType<Expression>(ind.getEnd(), table); error)
			return error;
	}

	return Error::success();
}

Error TypeChecker::checkType(Equation& eq, const SymbolTable& table)
{
	auto& lhs = eq.getLeftHand();
	auto& rhs = eq.getRightHand();

	if (auto error = checkType<Expression>(lhs, table); error)
		return error;

	if (auto error = checkType<Expression>(rhs, table); error)
		return error;

	auto& lhsType = lhs.getType();
	auto& rhsType = rhs.getType();

	if (lhs.isA<Tuple>() && lhs.get<Tuple>().size() > 1)
		if (!rhsType.isA<UserDefinedType>() ||
				lhs.get<Tuple>().size() > rhsType.get<UserDefinedType>().size())
			return make_error<IncompatibleType>("Type dimension mismatch");

	if (lhs.isA<Tuple>())
	{
		auto& tuple = lhs.get<Tuple>();
		auto& types = rhsType.get<UserDefinedType>();

		// Assign type to dummy variables.
		// The assignment can't be done earlier because the expression type would
		// have not been evaluated yet.

		for (size_t i = 0; i < tuple.size(); i++)
		{
			// If it's not a direct reference access, there's no way it can be a
			// dummy variable.
			if (!tuple[i]->isA<ReferenceAccess>())
				continue;

			auto& ref = tuple[i]->get<ReferenceAccess>();

			if (ref.isDummy())
			{
				assert(types.size() >= i);
				tuple[i]->setType(types[i]);
			}
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (rhsType.isA<UserDefinedType>())
	{
		auto& types = rhsType.get<UserDefinedType>();
		size_t returns = types.size();

		vector<Expression> newDestinations;

		if (lhs.isA<Tuple>())
			for (auto& destination : lhs.get<Tuple>())
				newDestinations.push_back(move(*destination));
		else
			newDestinations.push_back(move(lhs));

		for (size_t i = newDestinations.size(); i < returns; i++)
			newDestinations.emplace_back(types[i], ReferenceAccess::dummy());

		Tuple tuple(move(newDestinations));
		eq.setLeftHand(Expression(rhsType, move(tuple)));
	}

	return Error::success();
}

Error TypeChecker::checkType(Statement& statement, const SymbolTable& table)
{
	return statement.visit(
			[&](auto& statement) { return checkType(statement, table); });
}

Error TypeChecker::checkType(
		AssignmentStatement& statement, const SymbolTable& table)
{
	auto destinations = statement.getDestinations();

	for (auto& destination : destinations)
	{
		if (auto error = checkType<Expression>(*destination, table); error)
			return error;

		// The destinations must be l-values.
		// The check can't be enforced at parsing time because the grammar
		// specifies the destinations as expressions.

		if (!destination->isLValue())
			return make_error<BadSemantic>(
					"Destinations of statements must be l-values");
	}

	auto& expression = statement.getExpression();

	if (auto error = checkType<Expression>(expression, table); error)
		return error;

	if (destinations.size() > 1 && !expression.getType().isA<UserDefinedType>())
		return make_error<IncompatibleType>(
				"The expression must return at least " +
				to_string(destinations.size()) + "values");

	// Assign type to dummy variables.
	// The assignment can't be done earlier because the expression type would
	// have not been evaluated yet.

	for (size_t i = 0; i < destinations.size(); i++)
	{
		// If it's not a direct reference access, there's no way it can be a
		// dummy variable.
		if (!destinations[i]->isA<ReferenceAccess>())
			continue;

		auto& ref = destinations[i]->get<ReferenceAccess>();

		if (ref.isDummy())
		{
			auto& expressionType = expression.getType();
			assert(expressionType.isA<UserDefinedType>());
			auto& userDefType = expressionType.get<UserDefinedType>();
			assert(userDefType.size() >= i);
			destinations[i]->setType(userDefType[i]);
		}
	}

	// If the function call has more return values than the provided
	// destinations, then we need to add more dummy references.

	if (expression.getType().isA<UserDefinedType>())
	{
		auto& userDefType = expression.getType().get<UserDefinedType>();
		size_t returns = userDefType.size();

		if (destinations.size() < returns)
		{
			vector<Expression> newDestinations;

			for (auto& destination : destinations)
				newDestinations.push_back(move(*destination));

			for (size_t i = newDestinations.size(); i < returns; i++)
				newDestinations.emplace_back(userDefType[i], ReferenceAccess::dummy());

			statement.setDestination(Tuple(move(newDestinations)));
		}
	}

	return Error::success();
}

Error TypeChecker::checkType(IfStatement& statement, const SymbolTable& table)
{
	for (auto& block : statement)
		if (auto error = checkType(block, table); error)
			return error;

	return Error::success();
}

Error TypeChecker::checkType(
		IfStatement::Block& block, const SymbolTable& table)
{
	if (auto error = checkType<Expression>(block.getCondition(), table); error)
		return error;

	for (auto& statement : block)
		if (auto error = checkType(*statement, table); error)
			return error;

	return Error::success();
}

Error TypeChecker::checkType(ForStatement& statement, const SymbolTable& table)
{
	auto& induction = statement.getInduction();

	if (auto error = checkType<Expression>(induction.getBegin(), table); error)
		return error;

	if (auto error = checkType<Expression>(induction.getEnd(), table); error)
		return error;

	for (auto& stmnt : statement)
		if (auto error = checkType(*stmnt, table); error)
			return error;

	return Error::success();
}

Error TypeChecker::checkType(BreakStatement& statement, const SymbolTable& table)
{
	return Error::success();
}

Error TypeChecker::checkType(ReturnStatement& statement, const SymbolTable& table)
{
	return Error::success();
}

static Error subscriptionCheckType(Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<Operation>());
	assert(exp.get<Operation>().getKind() == OperationKind::subscription);

	auto& op = exp.get<Operation>();
	size_t subscriptionIndicesCount = op.argumentsCount() - 1;

	if (subscriptionIndicesCount > op[0].getType().dimensionsCount())
		return make_error<IncompatibleType>("array was subscripted too many times");

	for (size_t a = 1; a < op.argumentsCount(); a++)
		if (op[a].getType() != makeType<int>())
			return make_error<IncompatibleType>(
					"parameter of array subscription was not int");

	exp.setType(op[0].getType().subscript(subscriptionIndicesCount));
	return Error::success();
}

template<>
Error TypeChecker::checkType<Expression>(
		Expression& exp, const SymbolTable& table)
{
	if (exp.isA<Operation>())
		return checkType<Operation>(exp, table);

	if (exp.isA<Constant>())
		return checkType<Constant>(exp, table);

	if (exp.isA<ReferenceAccess>())
		return checkType<ReferenceAccess>(exp, table);

	if (exp.isA<Call>())
		return checkType<Call>(exp, table);

	if (exp.isA<Tuple>())
		return checkType<Tuple>(exp, table);

	assert(false && "Unreachable");
	return Error::success();
}

template<>
Error TypeChecker::checkType<Operation>(
		Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<Operation>());
	auto& op = exp.get<Operation>();

	for (auto& arg : op)
		if (auto error = checkType<Expression>(arg, table); error)
			return error;

	switch (op.getKind())
	{
		case OperationKind::negate:
		case OperationKind::add:
		case OperationKind::subtract:
		case OperationKind::multiply:
		case OperationKind::divide:
		case OperationKind::powerOf:
			exp.setType(op[0].getType());
			return Error::success();

		case OperationKind::ifelse:
			if (op[0].getType() != makeType<bool>())
				return make_error<IncompatibleType>(
						"condition of if else was not boolean");
			if (op[1].getType() != op[2].getType())
				return make_error<IncompatibleType>(
						"ternary operator branches had different return type");

			exp.setType(op[1].getType());
			return Error::success();

		case OperationKind::greater:
		case OperationKind::greaterEqual:
		case OperationKind::equal:
		case OperationKind::different:
		case OperationKind::lessEqual:
		case OperationKind::less:
			exp.setType(makeType<bool>());
			return Error::success();

		case OperationKind::lor:
		case OperationKind::land:
			if (op[0].getType() != makeType<bool>())
				return make_error<IncompatibleType>(
						"boolean operator had non boolean argument");
			if (op[1].getType() != makeType<bool>())
				return make_error<IncompatibleType>(
						"boolean operator had non boolean argument");
			exp.setType(makeType<bool>());
			return Error::success();

		case OperationKind::subscription:
			return subscriptionCheckType(exp, table);

		case OperationKind::memberLookup:
			return make_error<NotImplemented>("member lookup is not implemented yet");
	}

	assert(false && "unreachable");
	return make_error<NotImplemented>("op was not any supported kind");
}

template<>
Error TypeChecker::checkType<Constant>(
		Expression& expression, const SymbolTable& table)
{
	assert(expression.isA<Constant>());
	return Error::success();
}

template<>
Error TypeChecker::checkType<ReferenceAccess>(
		Expression& expression, const SymbolTable& table)
{
	assert(expression.isA<ReferenceAccess>());
	auto tp = typeFromSymbol(expression, table);

	if (!tp)
		return tp.takeError();

	expression.setType(move(*tp));
	return Error::success();
}

template<>
Error TypeChecker::checkType<Call>(
		Expression& expression, const SymbolTable& table)
{
	assert(expression.isA<Call>());
	auto& call = expression.get<Call>();

	for (size_t t : irange(call.argumentsCount()))
		if (auto error = checkType<Expression>(call[t], table); error)
			return error;

	auto& function = call.getFunction();

	if (auto error = checkType<Expression>(function, table); error)
		return error;

	if (function.get<ReferenceAccess>().getName() == "der")
		function.setType(call[0].getType());

	expression.setType(function.getType());

	return Error::success();
}

template<>
Error TypeChecker::checkType<Tuple>(
		Expression& expression, const SymbolTable& table)
{
	assert(expression.isA<Tuple>());
	auto& tuple = expression.get<Tuple>();

	SmallVector<Type, 3> types;

	for (const auto& exp : tuple)
	{
		if (auto error = checkType<Expression>(*exp, table); error)
			return error;

		types.push_back(exp->getType());
	}

	expression.setType(Type(types));
	return Error::success();
}

template<class T>
string getTemporaryVariableName(T& cls)
{
	const auto& members = cls.getMembers();
	int counter = 0;

	while (members.end() !=
				 find_if(members.begin(), members.end(), [=](const Member& obj) {
					 return obj.getName() == "_temp" + to_string(counter);
				 }))
		counter++;

	return "_temp" + to_string(counter);
}

llvm::Error resolveDummyReferences(Function& function)
{
	for (auto& algorithm : function.getAlgorithms())
	{
		for (auto& statement : algorithm.getStatements())
		{
			for (auto& assignment : statement)
			{
				for (auto& destination : assignment.getDestinations())
				{
					if (!destination->isA<ReferenceAccess>())
						continue;

					auto& ref = destination->get<ReferenceAccess>();

					if (!ref.isDummy())
						continue;

					string name = getTemporaryVariableName(function);
					Member temp(name, destination->getType(), TypePrefix::none());
					ref.setName(temp.getName());
					function.addMember(temp);

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return Error::success();
}

llvm::Error resolveDummyReferences(Class& model)
{
	for (auto& equation : model.getEquations())
	{
		auto& lhs = equation.getLeftHand();

		if (lhs.isA<Tuple>())
		{
			for (auto& expression : lhs.get<Tuple>())
			{
				if (!expression->isA<ReferenceAccess>())
					continue;

				auto& ref = expression->get<ReferenceAccess>();

				if (!ref.isDummy())
					continue;

				string name = getTemporaryVariableName(model);
				Member temp(name, expression->getType(), TypePrefix::none());
				ref.setName(temp.getName());
				model.addMember(temp);
			}
		}
	}

	// TODO: check of ForEquation

	for (auto& algorithm : model.getAlgorithms())
	{
		for (auto& statement : algorithm.getStatements())
		{
			for (auto& assignment : statement)
			{
				for (auto& destination : assignment.getDestinations())
				{
					if (!destination->isA<ReferenceAccess>())
						continue;

					auto& ref = destination->get<ReferenceAccess>();

					if (!ref.isDummy())
						continue;

					string name = getTemporaryVariableName(model);
					Member temp(name, destination->getType(), TypePrefix::none());
					ref.setName(temp.getName());
					model.addMember(temp);

					// Note that there is no need to add the dummy variable to the
					// symbol table, because it will never be referenced.
				}
			}
		}
	}

	return Error::success();
}
