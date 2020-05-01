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
	for (auto& ind : eq.getInductions())
	{
		if (auto error = fold(ind.getBegin(), table); error)
			return error;
		if (auto error = fold(ind.getEnd(), table); error)
			return error;
	}
	if (auto error = fold(eq.getEquation(), table); error)
		return error;

	return Error::success();
}

Error ConstantFolder::fold(Call& call, const SymbolTable& table)
{
	for (auto index : irange(call.argumentsCount()))
		if (auto error = fold(call[index], table); !error)
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
		if (auto error = fold(m, t); !error)
			return error;

	for (auto& eq : cl.getEquations())
		if (auto error = fold(eq, t); !error)
			return error;

	for (auto& eq : cl.getForEquations())
		if (auto error = fold(eq, t); !error)
			return error;

	return Error::success();
}
Error ConstantFolder::foldReference(Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<ReferenceAccess>());
	auto& ref = exp.get<ReferenceAccess>();

	const auto& simbol = table[ref.getName()].get<Member>();

	if (!simbol.hasInitializer())
		return Error::success();

	if (simbol.getInitializer().isA<Constant>() && simbol.isParameter())
		exp = simbol.getInitializer();
	return Error::success();
}

template<typename Type>
static Error foldOpSum(Expression& exp, SmallVector<Expression, 3> arguments)
{
	Expression::Operation::Container newArgs;

	Type val = 0;
	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val += arg.getConstant().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	if (newArgs.empty())
	{
		exp = Expression(makeType<Type>(), val);
		return Error::success();
	}

	arguments.emplace_back(makeType<Type>(), val);
	exp = Expression::op<OperationKind::add>(exp.getType(), move(newArgs));
	return Error::success();
}

template<typename Type>
static Error foldOpMult(Expression& exp, SmallVector<Expression, 3> arguments)
{
	Expression::Operation::Container newArgs;

	Type val = 1;
	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val *= arg.getConstant().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	if (newArgs.empty())
	{
		exp = Expression(makeType<Type>(), val);
		return Error::success();
	}

	arguments.emplace_back(makeType<Type>(), val);
	exp = Expression::op<OperationKind::add>(exp.getType(), move(newArgs));
	return Error::success();
}

template<typename Type>
static Error foldOpNegate(Expression& exp, SmallVector<Expression, 3> arguments)
{
	if (arguments[0].isA<Constant>())
		exp = Expression(
				arguments[0].getType(), -arguments[0].getConstant().get<Type>());
	return Error::success();
}

template<typename Type>
static Error foldOpSubtract(
		Expression& exp, SmallVector<Expression, 3> arguments)
{
	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		exp = Expression(
				arguments[0].getType(),
				arguments[0].getConstant().get<Type>() -
						arguments[1].getConstant().as<Type>());
	return Error::success();
}

template<typename Type>
static Error foldOpDivide(Expression& exp, SmallVector<Expression, 3> arguments)
{
	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		exp = Expression(
				arguments[0].getType(),
				arguments[0].getConstant().get<Type>() /
						arguments[1].getConstant().as<Type>());
	return Error::success();
}

static Error foldOpSum(Expression& exp, ArrayRef<Expression> arguments)
{
	auto argCopy = SmallVector<Expression, 3>(arguments.begin(), arguments.end());
	if (arguments[0].getType() == makeType<int>())
		return foldOpSum<int>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<float>())
		return foldOpSum<float>(exp, move(argCopy));
	return Error::success();
}

static Error foldOpSubtract(Expression& exp, ArrayRef<Expression> arguments)
{
	auto argCopy = SmallVector<Expression, 3>(arguments.begin(), arguments.end());
	if (arguments[0].getType() == makeType<int>())
		return foldOpSubtract<int>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<int>())
		return foldOpSubtract<float>(exp, move(argCopy));
	return Error::success();
}

static Error foldOpDivide(Expression& exp, ArrayRef<Expression> arguments)
{
	auto argCopy = SmallVector<Expression, 3>(arguments.begin(), arguments.end());
	if (arguments[0].getType() == makeType<int>())
		return foldOpDivide<int>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<float>())
		return foldOpDivide<float>(exp, move(argCopy));
	return Error::success();
}

static Error foldOpMult(Expression& exp, ArrayRef<Expression> arguments)
{
	auto argCopy = SmallVector<Expression, 3>(arguments.begin(), arguments.end());
	if (arguments[0].getType() == makeType<int>())
		return foldOpMult<int>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<float>())
		return foldOpMult<float>(exp, move(argCopy));
	return Error::success();
}

static Error foldOpNegate(Expression& exp, ArrayRef<Expression> arguments)
{
	auto argCopy = SmallVector<Expression, 3>(arguments.begin(), arguments.end());
	if (arguments[0].getType() == makeType<int>())
		return foldOpNegate<int>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<float>())
		return foldOpNegate<float>(exp, move(argCopy));
	if (arguments[0].getType() == makeType<bool>())
		return foldOpNegate<bool>(exp, move(argCopy));
	assert(false && "unrechable");
	return Error::success();
}

Error ConstantFolder::foldExpression(Expression& exp, const SymbolTable& table)
{
	assert(exp.isOperation());
	auto& op = exp.getOperation();
	for (auto& arg : op)
		if (auto error = fold(arg, table); error)
			return error;

	switch (op.getKind())
	{
		case OperationKind::negate:
			return foldOpNegate(exp, op.getArguments());
		case OperationKind::add:
			return foldOpSum(exp, op.getArguments());
		case OperationKind::subtract:
			return foldOpSubtract(exp, op.getArguments());
		case OperationKind::multiply:
			return foldOpMult(exp, op.getArguments());
		case OperationKind::divide:
			return foldOpDivide(exp, op.getArguments());
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
			return Error::success();
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
		return foldExpression(exp, table);
	assert(false && "unreachable");
	return make_error<NotImplemented>("found a not handled type of expression");
}
