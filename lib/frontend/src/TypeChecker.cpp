#include "modelica/frontend/TypeChecker.hpp"

#include <cstdio>

#include "llvm/Support/Error.h"
#include "modelica/frontend/Call.hpp"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/ForEquation.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/ParserErrors.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"
#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

Error TypeChecker::checkType(Class& cl, const SymbolTable& table)
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

Error TypeChecker::checkType(Member& mem, const SymbolTable& table)
{
	if (!mem.hasInitializer())
		return Error::success();
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

Error TypeChecker::checkCall(Expression& callExp, const SymbolTable& table)
{
	assert(callExp.isA<Call>());

	auto& call = callExp.get<Call>();
	for (size_t t : irange(call.argumentsCount()))
		if (auto error = checkType(call[t], table); error)
			return error;

	if (auto error = checkType(call.getFunction(), table); error)
		return error;

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

static Expected<Type> typeFromSymbol(
		const Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<ReferenceAccess>());
	ReferenceAccess acc = exp.get<ReferenceAccess>();
	const auto& name = acc.getName();
	if (name == "der")
		return Type::unkown();
	if (!table.hasSymbol(name))
		return make_error<NotImplemented>("no known variable named " + name);

	const auto& symbol = table[name];
	if (symbol.isA<Member>())
		return symbol.get<Member>().getType();

	if (symbol.isA<Induction>())
		return makeType<int>();

	return make_error<NotImplemented>("no known variable named " + name);
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
