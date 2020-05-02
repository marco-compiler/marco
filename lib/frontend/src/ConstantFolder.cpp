#include "modelica/frontend/ConstantFolder.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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
#include "modelica/utils/IRange.hpp"

using namespace llvm;
using namespace std;
using namespace modelica;

Error ConstantFolder::fold(Equation& eq, const SymbolTable& table)
{
	if (auto error = fold(eq.getLeftHand(), table); error)
		return error;
	if (auto error = fold(eq.getRightHand(), table); error)
		return error;

	return Error::success();
}

Error ConstantFolder::fold(ForEquation& eq, const SymbolTable& table)
{
	SymbolTable t(&table);
	for (auto& ind : eq.getInductions())
	{
		t.addSymbol(ind);
		if (auto error = fold(ind.getBegin(), table); error)
			return error;
		if (auto error = fold(ind.getEnd(), table); error)
			return error;
	}

	if (auto error = fold(eq.getEquation(), t); error)
		return error;

	return Error::success();
}

Error ConstantFolder::fold(Call& call, const SymbolTable& table)
{
	for (auto index : irange(call.argumentsCount()))
		if (auto error = fold(call[index], table); error)
			return error;

	return fold(call.getFunction(), table);
}

Error ConstantFolder::fold(Member& mem, const SymbolTable& table)
{
	if (mem.hasInitializer())
		return fold(mem.getInitializer(), table);
	return Error::success();
}

Error ConstantFolder::fold(Class& cl, const SymbolTable& table)
{
	SymbolTable t(cl, &table);

	for (auto& m : cl.getMembers())
		if (auto error = fold(m, t); error)
			return error;

	for (auto& eq : cl.getEquations())
		if (auto error = fold(eq, t); error)
			return error;

	for (auto& eq : cl.getForEquations())
		if (auto error = fold(eq, t); error)
			return error;

	return Error::success();
}
Error ConstantFolder::foldReference(Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<ReferenceAccess>());
	auto& ref = exp.get<ReferenceAccess>();

	if (!table.hasSymbol(ref.getName()))
		return Error::success();
	const auto& s = table[ref.getName()];
	if (!s.isA<Member>())
		return Error::success();
	const auto simbol = s.get<Member>();

	if (!simbol.isParameter())
		return Error::success();

	if (!simbol.hasInitializer())
		return Error::success();

	if (simbol.getInitializer().isA<Constant>() && simbol.isParameter())
		exp = simbol.getInitializer();
	return Error::success();
}

using Vector = Expression::Operation::Container;

template<typename Type>
static Expected<Expression> foldOpSum(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	Vector newArgs;

	Type val = 0;
	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val += arg.getConstant().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	if (newArgs.empty())
		return Expression(makeType<Type>(), val);

	if (val != 0)
		newArgs.push_back(Expression(makeType<Type>(), val));

	if (newArgs.size() == 1)
		return newArgs[0];

	return Expression::op<OperationKind::add>(exp.getType(), move(newArgs));
}

template<typename Type>
static Expected<Expression> foldOpMult(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	Vector newArgs;

	Type val = 1;
	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val *= arg.getConstant().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	if (newArgs.empty())
		return Expression(makeType<Type>(), val);

	if (val != 1)
		newArgs.push_back(Expression(makeType<Type>(), val));

	if (newArgs.size() == 1)
		return newArgs[0];

	return Expression::op<OperationKind::multiply>(exp.getType(), move(newArgs));
}

template<typename Type>
static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].isA<Constant>())
		return Expression(
				arguments[0].getType(), -arguments[0].getConstant().get<Type>());
	return exp;
}

template<typename Type>
static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments.size() == 1)
		if (arguments[0].isA<Constant>())
			return Expression(
					arguments[0].getType(), -arguments[0].getConstant().get<Type>());

	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression(
				arguments[0].getType(),
				arguments[0].getConstant().as<Type>() -
						arguments[1].getConstant().as<Type>());
	return exp;
}

template<typename Type>
static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression(
				arguments[0].getType(),
				arguments[0].getConstant().as<Type>() /
						arguments[1].getConstant().as<Type>());
	return exp;
}

static Expected<Expression> foldOpSum(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].getType() == makeType<int>())
		return foldOpSum<int>(exp);
	if (arguments[0].getType() == makeType<float>())
		return foldOpSum<float>(exp);
	return exp;
}

static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].getType() == makeType<int>())
		return foldOpSubtract<int>(exp);
	if (arguments[0].getType() == makeType<int>())
		return foldOpSubtract<float>(exp);

	return exp;
}

static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].getType() == makeType<int>())
		return foldOpDivide<int>(exp);
	if (arguments[0].getType() == makeType<float>())
		return foldOpDivide<float>(exp);
	return exp;
}

static Expected<Expression> foldOpMult(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].getType() == makeType<int>())
		return foldOpMult<int>(exp);
	if (arguments[0].getType() == makeType<float>())
		return foldOpMult<float>(exp);
	return exp;
}

static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.getOperation().getArguments();
	if (arguments[0].getType() == makeType<int>())
		return foldOpNegate<int>(exp);
	if (arguments[0].getType() == makeType<float>())
		return foldOpNegate<float>(exp);
	if (arguments[0].getType() == makeType<bool>())
		return foldOpNegate<bool>(exp);
	assert(false && "unrechable");
	return exp;
}

Expected<Expression> ConstantFolder::foldExpression(
		Expression& exp, const SymbolTable& table)
{
	assert(exp.isOperation());
	auto& op = exp.getOperation();
	for (auto& arg : op)
		if (auto error = fold(arg, table); error)
			return move(error);

	switch (op.getKind())
	{
		case OperationKind::negate:
			return foldOpNegate(exp);
		case OperationKind::add:
			return foldOpSum(exp);
		case OperationKind::subtract:
			return foldOpSubtract(exp);
		case OperationKind::multiply:
			return foldOpMult(exp);
		case OperationKind::divide:
			return foldOpDivide(exp);
		case OperationKind::ifelse:
		case OperationKind::greater:
		case OperationKind::greaterEqual:
		case OperationKind::equal:
		case OperationKind::different:
		case OperationKind::lessEqual:
		case OperationKind::less:
		case OperationKind::land:
		case OperationKind::lor:
		case OperationKind::subscription:
		case OperationKind::memberLookup:
		case OperationKind::powerOf:
			return exp;
	}

	assert(false && "unrechable");
	return make_error<NotImplemented>("found a not handled kind of operation");
}

Error ConstantFolder::fold(Expression& exp, const SymbolTable& table)
{
	if (exp.isA<Constant>())
		return Error::success();
	if (exp.isA<Call>())
		return fold(exp.get<Call>(), table);
	if (exp.isA<ReferenceAccess>())
		return foldReference(exp, table);
	if (exp.isA<Expression::Operation>())
	{
		auto newexp = foldExpression(exp, table);
		if (!newexp)
			return newexp.takeError();
		exp = move(*newexp);
		return Error::success();
	}
	assert(false && "unreachable");
	return make_error<NotImplemented>("found a not handled type of expression");
}
