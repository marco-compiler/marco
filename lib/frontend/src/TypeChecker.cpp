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
#include <stack>

using namespace modelica;
using namespace llvm;
using namespace std;

static Expected<Type> typeFromSymbol(
		const Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<ReferenceAccess>());
	ReferenceAccess acc = exp.get<ReferenceAccess>();
	const auto& name = acc.getName();

	if (name == "der")
		return Type::unknown();

	if (!table.hasSymbol(name))
		return make_error<NotImplemented>("No known variable named " + name);

	const auto& symbol = table[name];

	// TODO: output type for functions
	if (symbol.isA<Class>())
		return Type::unknown();

	if (symbol.isA<Member>())
		return symbol.get<Member>().getType();

	if (symbol.isA<Induction>())
		return makeType<int>();

	return make_error<NotImplemented>("No known variable named " + name);
}

Error TypeChecker::checkType(Algorithm& algorithm, const SymbolTable& table)
{
	for (auto& statement : algorithm.getStatements())
		if (auto error = checkType(statement, table); error)
			return error;

	return Error::success();
}

template<>
Error TypeChecker::checkType<ClassType::Class>(
		Class& cl, const SymbolTable& table)
{
	SymbolTable t(cl, &table);

	for (auto& m : cl.getMembers())
		if (auto error = checkType(m, t); error)
			return error;

	for (auto& eq : cl.getEquations())
		if (auto error = checkType(eq, t); error)
			return error;

	for (auto& eq : cl.getForEquations())
		if (auto error = checkType(eq, t); error)
			return error;

	return Error::success();
}

template<>
Error TypeChecker::checkType<ClassType::Function>(
		Class& cl, const SymbolTable& table)
{
	SymbolTable t(cl, &table);

	for (auto& member : cl.getMembers())
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
	}

	auto& algorithms = cl.getAlgorithms();

	// From Function reference:
	// "A function can have at most one algorithm section or one external
	// function interface (not both), which, if present, is the body of the
	// function."

	if (algorithms.size() == 1)
		return make_error<BadSemantic>(
				"Functions must have exactly one algorithm section");

	for (auto& statement : algorithms[0].getStatements())
	{
		if (auto error = checkType(statement, t); error)
			return error;

		for (auto& destination : statement.getDestinations())
		{
			if (auto error = checkType(destination, t); error)
				return error;

			// From Function reference:
			// "Input formal parameters are read-only after being bound to the actual
			// arguments or default values, i.e., they may not be assigned values in
			// the body of the function."

			auto& exp = destination;

			while (exp.isOperation())
			{
				auto& operation = exp.getOperation();
				assert(operation.getKind() == OperationKind::subscription);
				exp = operation[0];
			}

			assert(exp.isA<ReferenceAccess>());
			auto& ref = exp.get<ReferenceAccess>();
			const auto& name = ref.getName();

			if (!t.hasSymbol(name))
				return make_error<NotImplemented>("No known variable named " + name);

			const auto& member = t[name].get<Member>();

			if (member.isInput())
				return make_error<BadSemantic>(
						"Input variables can't receive a new value");
		}

		// From Function reference:
		// "A function cannot contain calls to the Modelica built-in operators der,
		// initial, terminal, sample, pre, edge, change, reinit, delay, cardinality,
		// inStream, actualStream, to the operators of the built-in package
		// Connections, and is not allowed to contain when-statements."

		stack<Expression> stack;
		stack.push(statement.getExpression());

		while (!stack.empty())
		{
			auto expression = stack.top();
			stack.pop();

			if (expression.isA<ReferenceAccess>())
			{
				string& name = expression.get<ReferenceAccess>().getName();

				if (name == "der" || name == "initial" || name == "terminal" ||
						name == "sample" || name == "pre" || name == "edge" ||
						name == "change" || name == "reinit" || name == "delay" ||
						name == "cardinality" || name == "inStream" ||
						name == "actualStream")
				{
					return make_error<BadSemantic>(
							name + " is not allowed in procedural code");
				}

				// TODO: Connections built-in operators + when statement
			}
			else if (expression.isOperation())
			{
				for (auto& arg : expression.getOperation())
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

	return Error::success();
}

template<>
Error TypeChecker::checkType<ClassType::Model>(
		Class& cl, const SymbolTable& table)
{
	return checkType<ClassType::Class>(cl, table);
}

Error TypeChecker::checkType(Member& mem, const SymbolTable& table)
{
	if (mem.hasInitializer())
		if (auto error = checkType(mem.getInitializer(), table); error)
			return error;

	if (not mem.hasStartOverload())
		return Error::success();

	if (auto error = checkType(mem.getStartOverload(), table); error)
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
		if (auto error = checkType(ind.getBegin(), table); error)
			return error;

		if (auto error = checkType(ind.getEnd(), table); error)
			return error;
	}

	return Error::success();
}

Error TypeChecker::checkType(Equation& eq, const SymbolTable& table)
{
	if (auto error = checkType(eq.getLeftHand(), table); error)
		return error;

	if (auto error = checkType(eq.getRightHand(), table); error)
		return error;

	return Error::success();
}

Error TypeChecker::checkType(Statement& statement, const SymbolTable& table)
{
	for (auto& destination : statement.getDestinations())
	{
		// The destinations must be l-values.
		// The check can't be enforced at parsing time because the grammar
		// specifies the destinations as expressions.

		if (!destination.isLValue())
			return make_error<BadSemantic>(
					"Destinations of statements must be l-values");
	}

	if (auto error = checkType(statement.getExpression(), table); error)
		return error;

	return Error::success();
}

Error TypeChecker::checkCall(Expression& callExp, const SymbolTable& table)
{
	assert(callExp.isA<Call>());

	auto& call = callExp.get<Call>();

	for (size_t t : irange(call.argumentsCount()))
		if (auto error = checkType(call[t], table); error)
			return error;

	if (auto error = checkType(call.getFunction(), table); error)
		return error;

	/*
	if (!call.getFunction().isA<ReferenceAccess>())
		return make_error<NotImplemented>("only der function is implemented");

	if (call.getFunction().get<ReferenceAccess>().getName() != "der")
		return make_error<NotImplemented>("only der function is implemented");

	if (call.argumentsCount() != 1)
		return make_error<NotImplemented>(
				"only der with one argument are supported");
				*/

	if (auto error = checkType(call[0], table); error)
		return error;

	callExp.setType(call[0].getType());
	return Error::success();
}

static Error subscriptionCheckType(Expression& exp, const SymbolTable& table)
{
	assert(exp.isOperation());
	assert(exp.getOperation().getKind() == OperationKind::subscription);

	auto& op = exp.getOperation();
	size_t subscriptionIndiciesCount = op.argumentsCount() - 1;

	if (subscriptionIndiciesCount > op[0].getType().dimensionsCount())
		return make_error<IncompatibleType>("array was subscripted too many times");

	for (size_t a = 1; a < op.argumentsCount(); a++)
		if (op[a].getType() != makeType<int>())
			return make_error<IncompatibleType>(
					"parameter of array subscription was not int");

	exp.setType(op[0].getType().subscript(subscriptionIndiciesCount));
	return Error::success();
}

Error TypeChecker::checkOperation(Expression& exp, const SymbolTable& table)
{
	assert(exp.isOperation());
	auto& op = exp.getOperation();

	for (auto& arg : op)
		if (auto error = checkType(arg, table); error)
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

Error TypeChecker::checkType(Expression& exp, const SymbolTable& table)
{
	if (exp.isA<Constant>())
		return Error::success();

	if (exp.isA<Call>())
		return checkCall(exp, table);

	if (exp.isA<ReferenceAccess>())
	{
		auto tp = typeFromSymbol(exp, table);
		if (!tp)
			return tp.takeError();
		exp.setType(move(*tp));
		return Error::success();
	}

	if (exp.isOperation())
	{
		return checkOperation(exp, table);
	}

	assert(false && "unreachable");
	return make_error<NotImplemented>("exp was not any supported type");
}
