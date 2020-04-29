#include "modelica/frontend/TypeChecker.hpp"

#include <cstdio>

#include "llvm/Support/Error.h"
#include "modelica/frontend/Call.hpp"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/ParserErrors.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Error TypeChecker::checkType(Class& cl, const SymbolTable& table)
{
	SymbolTable t(cl, &table);
	for (auto& m : cl.getMembers())
	{
		if (auto error = checkType(m, t); error)
			return error;
	}

	for (auto& eq : cl.getEquations())
	{
		if (auto error = checkType(eq, t); error)
			return error;
	}
	return Error::success();
}

Error TypeChecker::checkType(Member& mem, const SymbolTable& table)
{
	if (!mem.hasInitializer())
		return Error::success();
	if (auto error = checkType(mem.getInitializer(), table); error)
		return error;

	if (mem.getInitializer().getType() != mem.getType())
		return make_error<IncompatibleType>(
				"type of " + mem.getName() +
				" initializer does not match the variable type");

	return Error::success();
}

Error TypeChecker::checkType(Equation& eq, const SymbolTable& table)
{
	if (auto error = checkType(eq.getLeftHand(), table); error)
		return error;
	if (auto error = checkType(eq.getRightHand(), table); error)
		return error;

	if (eq.getLeftHand().getType() != eq.getRightHand().getType())
		return make_error<IncompatibleType>("type missmatch in equation");
	return Error::success();
}

Error TypeChecker::checkCall(Expression& callExp, const SymbolTable& table)
{
	assert(callExp.isA<Call>());
	auto& call = callExp.get<Call>();
	if (!call.getFunction().isA<ReferenceAccess>())
		return make_error<NotImplemented>("only der function is implemented");

	if (call.getFunction().get<ReferenceAccess>().getName() != "der")
		return make_error<NotImplemented>("only der function is implemented");

	if (call.argumentsCount() != 1)
		return make_error<NotImplemented>(
				"only der with one argument are supported");

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
		const auto& name = exp.get<ReferenceAccess>().getName();
		if (!table.hasSymbol(exp.get<ReferenceAccess>().getName()))
			return make_error<NotImplemented>("no known variable named " + name);

		const auto& symbol = table[name];
		if (!symbol.isA<Member>())
			return make_error<NotImplemented>("no known variable named " + name);

		exp.setType(symbol.get<Member>().getType());
		return Error::success();
	}
	if (exp.isOperation())
	{
		return checkOperation(exp, table);
	}

	assert(false && "unreachable");
	return make_error<NotImplemented>("exp was not any supported type");
}
