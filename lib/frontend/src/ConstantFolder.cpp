#include <iterator>
#include <limits>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Constant.hpp>
#include <modelica/frontend/ConstantFolder.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Member.hpp>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/ReferenceAccess.hpp>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

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
		if (auto error = fold(mem.getInitializer(), table); error)
			return error;

	if (mem.hasStartOverload())
		return fold(mem.getStartOverload(), table);

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

/**
 * tries to fold references of known variable that have a initializer
 */
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

	if (!simbol.hasInitializer())
		return Error::success();

	if (simbol.getInitializer().isA<Constant>() && simbol.isParameter())
		exp = simbol.getInitializer();

	return Error::success();
}

using Vector = Operation::Container;

template<typename IteratorFrom, typename IteratorTo>
void moveRange(IteratorFrom b, IteratorFrom e, IteratorTo b2)
{
	move(make_move_iterator(b), make_move_iterator(e), b2);
}

/**
 * Given a associative operation kind and a expression,
 * it brings all nested expressions to the top level.
 *
 * As an example a + (b+(2*c)) becomes a + b + (2*c)
 */
static void flatten(Expression& exp, OperationKind kind)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	SmallVector<Expression, 3> newArguments;

	for (size_t a = arguments.size() - 1; a != numeric_limits<size_t>::max(); a--)
	{
		// if a argument is not a operation we cannot do anything
		if (!arguments[a].isA<Operation>())
			continue;

		// if it is not the operation we are looking for we leave it there
		auto& op = arguments[a].get<Operation>();
		if (op.getKind() != kind)
			continue;

		// otherwise we take all its arguments and we place them in the
		moveRange(
				op.getArguments().begin(),
				op.getArguments().end(),
				back_inserter(newArguments));

		// finally we remove that argument from the list
		arguments.erase(arguments.begin() + a);
	}

	// all new arguments must be added
	moveRange(newArguments.begin(), newArguments.end(), back_inserter(arguments));
}

template<BuiltinType Type>
static Expected<Expression> foldOpSum(Expression& exp)
{
	flatten(exp, OperationKind::add);
	Vector& arguments = exp.get<Operation>().getArguments();
	Vector newArgs;

	using Tr = frontendTypeToType_v<Type>;

	// for each argument, if it is a constant add it to to val, otherwise add it
	// to the new arguments.
	Tr val = 0;

	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val += arg.get<Constant>().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	// if there are not args left transform it into a constant
	if (newArgs.empty())
		return Expression(makeType<Tr>(), val);

	// if the sum of constants is not zero insert a new constant argument
	if (val != 0)
		newArgs.push_back(Expression(makeType<Tr>(), val));

	// if the arguments are exactly one, remove the sum and return the argument
	// itself
	if (newArgs.size() == 1)
		return newArgs[0];

	// other wise return the sum.
	return Expression::add(exp.getType(), move(newArgs));
}

template<BuiltinType Type>
static Expected<Expression> foldOpMult(Expression& exp)
{
	// works as the sum, read that one.
	flatten(exp, OperationKind::multiply);
	Vector& arguments = exp.get<Operation>().getArguments();
	Vector newArgs;

	using Tr = frontendTypeToType_v<Type>;
	Tr val = 1;
	for (auto& arg : arguments)
		if (arg.isA<Constant>())
			val *= arg.get<Constant>().as<Type>();
		else
			newArgs.emplace_back(move(arg));

	if (newArgs.empty())
		return Expression(makeType<Tr>(), val);

	if (val != 1)
		newArgs.push_back(Expression(makeType<Tr>(), val));

	if (newArgs.size() == 1)
		return newArgs[0];

	return Expression::multiply(exp.getType(), move(newArgs));
}

template<BuiltinType Type>
static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments[0].isA<Constant>())
		return Expression(
				arguments[0].getType(), -arguments[0].get<Constant>().get<Type>());
	return exp;
}

template<BuiltinType Type>
static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments.size() == 1)
		if (arguments[0].isA<Constant>())
			return Expression(
					arguments[0].getType(), -arguments[0].get<Constant>().get<Type>());

	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression(
				arguments[0].getType(),
				arguments[0].get<Constant>().as<Type>() -
						arguments[1].get<Constant>().as<Type>());
	return exp;
}

template<BuiltinType Type>
static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression(
				arguments[0].getType(),
				arguments[0].get<Constant>().as<Type>() /
						arguments[1].get<Constant>().as<Type>());
	return exp;
}

static Expected<Expression> foldOpSum(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpSum<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpSum<BuiltinType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpSubtract<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpSubtract<BuiltinType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpDivide<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpDivide<BuiltinType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpMult(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpMult<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpMult<BuiltinType::Float>(exp);

	return exp;
}

template<BuiltinType T>
static Expected<Expression> foldOpExp(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	assert(arguments.size() == 2);

	if (!arguments[0].isA<Constant>() || !arguments[1].isA<Constant>())
		return exp;

	using Tr = frontendTypeToType_v<T>;
	Tr val =
			pow(arguments[0].get<Constant>().as<T>(),
					arguments[1].get<Constant>().as<T>());

	return Expression(makeType<Tr>(), val);
}

static Expected<Expression> foldOpExp(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpExp<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpExp<BuiltinType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltinType::Integer>())
		return foldOpNegate<BuiltinType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltinType::Float>())
		return foldOpNegate<BuiltinType::Float>(exp);

	if (arguments[0].getType() == makeType<bool>())
		return foldOpNegate<BuiltinType::Boolean>(exp);

	assert(false && "unrechable");
	return exp;
}

static Expected<Expression> foldOpSubscrition(
		Expression& exp, const SymbolTable& table)
{
	return exp;
	assert(exp.get<Operation>().getKind() == OperationKind::subscription);

	const auto& target = exp.get<Operation>().getArguments()[0];
	if (not exp.getType().isScalar())
		return exp;

	if (not target.isA<ReferenceAccess>())
		return exp;

	const auto& vName = target.get<ReferenceAccess>().getName();

	if (not table.hasSymbol(vName))
		return exp;

	const auto& e = table[vName];
	if (not e.isA<Member>())
		return exp;

	const auto& m = e.get<Member>();
	if (not m.isParameter())
		return exp;

	if (m.hasInitializer())
		return exp;

	if (not m.hasStartOverload())
		return exp;

	return m.getStartOverload();
}

Expected<Expression> ConstantFolder::foldExpression(
		Expression& exp, const SymbolTable& table)
{
	assert(exp.isA<Operation>());
	auto& op = exp.get<Operation>();
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
		case OperationKind::powerOf:
			return foldOpExp(exp);
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
			return foldOpSubscrition(exp, table);
		case OperationKind::memberLookup:
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

	if (exp.isA<Operation>())
	{
		auto newexp = foldExpression(exp, table);

		if (!newexp)
			return newexp.takeError();

		exp = move(*newexp);
		assert(
				exp.isA<Call>() or
				exp.getType().getBuiltIn() != Type::unknown().getBuiltIn());
		return Error::success();
	}

	assert(false && "unreachable");
	return make_error<NotImplemented>("found a not handled type of expression");
}
