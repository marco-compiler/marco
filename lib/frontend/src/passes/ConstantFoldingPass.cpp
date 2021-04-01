#include <iterator>
#include <limits>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/passes/ConstantFoldingPass.h>
#include <modelica/frontend/ParserErrors.hpp>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica::frontend;
using namespace std;

Error ConstantFolder::run(Equation& equation)
{
	if (auto error = run(equation.getLeftHand()); error)
		return error;

	if (auto error = run(equation.getRightHand()); error)
		return error;

	return Error::success();
}

Error ConstantFolder::run(ForEquation& forEquation)
{
	SymbolTableScope varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind.getName(), Symbol(ind));

		if (auto error = run(ind.getBegin()); error)
			return error;

		if (auto error = run(ind.getEnd()); error)
			return error;
	}

	if (auto error = run(forEquation.getEquation()); error)
		return error;

	return Error::success();
}

Error ConstantFolder::run(Call& call)
{
	for (auto index : irange(call.argumentsCount()))
		if (auto error = run(call[index]); error)
			return error;

	return run(call.getFunction());
}

Error ConstantFolder::run(Member& member)
{
	if (member.hasInitializer())
		if (auto error = run(member.getInitializer()); error)
			return error;

	if (member.hasStartOverload())
		return run(member.getStartOverload());

	return Error::success();
}

Error ConstantFolder::run(ClassContainer& cls)
{
	return cls.visit([&](auto& obj) { return run(obj); });
}

Error ConstantFolder::run(Class& cls)
{
	SymbolTableScope varScope(symbolTable);

	// Populate the symbol table
	symbolTable.insert(cls.getName(), Symbol(cls));

	for (auto& member : cls.getMembers())
		symbolTable.insert(member->getName(), Symbol(*member));

	for (auto& m : cls.getMembers())
		if (auto error = run(*m); error)
			return error;

	for (auto& eq : cls.getEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& eq : cls.getForEquations())
		if (auto error = run(*eq); error)
			return error;

	return Error::success();
}

Error ConstantFolder::run(Function& function)
{
	SymbolTableScope varScope(symbolTable);

	// Populate the symbol table
	symbolTable.insert(function.getName(), Symbol(function));

	for (auto& member : function.getMembers())
		if (auto error = run(*member); error)
			return error;



	return Error::success();
}

Error ConstantFolder::run(Package& package)
{
	for (auto& cls : package)
		if (auto error = run(cls); error)
			return error;

	return Error::success();
}

Error ConstantFolder::run(Record& record)
{
	return Error::success();
}

/**
 * tries to fold references of known variable that have a initializer
 */
Error ConstantFolder::foldReference(Expression& expression)
{
	assert(expression.isA<ReferenceAccess>());
	auto& ref = expression.get<ReferenceAccess>();

	if (symbolTable.count(ref.getName()) == 0)
		return Error::success();

	const auto& s = symbolTable.lookup(ref.getName());
	if (!s.isA<Member>())
		return Error::success();

	const auto symbol = s.get<Member>();

	if (!symbol.hasInitializer())
		return Error::success();

	if (symbol.getInitializer().isA<Constant>() && symbol.isParameter())
		expression = symbol.getInitializer();

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

template<BuiltInType Type>
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
		return Expression::constant(exp.getLocation(), makeType<Tr>(), val);

	// if the sum of constants is not zero insert a new constant argument
	if (val != 0)
		newArgs.push_back(Expression::constant(exp.getLocation(), makeType<Tr>(), val));

	// if the arguments are exactly one, remove the sum and return the argument
	// itself
	if (newArgs.size() == 1)
		return newArgs[0];

	// other wise return the sum.
	return Expression::operation(exp.getLocation(), exp.getType(), OperationKind::add, move(newArgs));
}

template<BuiltInType Type>
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
		return Expression::constant(exp.getLocation(), makeType<Tr>(), val);

	if (val != 1)
		newArgs.push_back(Expression::constant(exp.getLocation(), makeType<Tr>(), val));

	if (newArgs.size() == 1)
		return newArgs[0];

	return Expression::operation(exp.getLocation(), exp.getType(), OperationKind::multiply, move(newArgs));
}

template<BuiltInType Type>
static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments[0].isA<Constant>())
		return Expression::constant(exp.getLocation(), arguments[0].getType(), -arguments[0].get<Constant>().get<Type>());
	return exp;
}

template<BuiltInType Type>
static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments.size() == 1)
		if (arguments[0].isA<Constant>())
			return Expression::constant(exp.getLocation(), arguments[0].getType(), -arguments[0].get<Constant>().get<Type>());

	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression::constant(
				exp.getLocation(),
				arguments[0].getType(),
				arguments[0].get<Constant>().as<Type>() -
				arguments[1].get<Constant>().as<Type>());
	return exp;
}

template<BuiltInType Type>
static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();
	if (arguments[0].isA<Constant>() && arguments[1].isA<Constant>())
		return Expression::constant(
				exp.getLocation(),
				arguments[0].getType(),
				arguments[0].get<Constant>().as<Type>() /
				arguments[1].get<Constant>().as<Type>());
	return exp;
}

static Expected<Expression> foldOpSum(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpSum<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpSum<BuiltInType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpSubtract(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpSubtract<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpSubtract<BuiltInType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpDivide(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpDivide<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpDivide<BuiltInType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpMult(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpMult<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpMult<BuiltInType::Float>(exp);

	return exp;
}

template<BuiltInType T>
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

	return Expression::constant(exp.getLocation(), makeType<Tr>(), val);
}

static Expected<Expression> foldOpExp(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpExp<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpExp<BuiltInType::Float>(exp);

	return exp;
}

static Expected<Expression> foldOpNegate(Expression& exp)
{
	Vector& arguments = exp.get<Operation>().getArguments();

	if (arguments[0].getType() == makeType<BuiltInType::Integer>())
		return foldOpNegate<BuiltInType::Integer>(exp);

	if (arguments[0].getType() == makeType<BuiltInType::Float>())
		return foldOpNegate<BuiltInType::Float>(exp);

	if (arguments[0].getType() == makeType<bool>())
		return foldOpNegate<BuiltInType::Boolean>(exp);

	assert(false && "unrechable");
	return exp;
}

Expected<Expression> ConstantFolder::foldOpSubscrition(Expression& exp)
{
	return exp;
	assert(exp.get<Operation>().getKind() == OperationKind::subscription);

	const auto& target = exp.get<Operation>().getArguments()[0];
	if (not exp.getType().isScalar())
		return exp;

	if (not target.isA<ReferenceAccess>())
		return exp;

	const auto& vName = target.get<ReferenceAccess>().getName();

	if (symbolTable.count(vName) == 0)
		return exp;

	const auto& e = symbolTable.lookup(vName);
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
		Expression& exp)
{
	assert(exp.isA<Operation>());
	auto& op = exp.get<Operation>();
	for (auto& arg : op)
		if (auto error = run(arg); error)
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
			return foldOpSubscrition(exp);
		case OperationKind::memberLookup:
			return exp;
	}

	assert(false && "unrechable");
	return make_error<NotImplemented>("found a not handled kind of operation");
}

Error ConstantFolder::run(Expression& expression)
{
	if (expression.isA<Constant>())
		return Error::success();

	if (expression.isA<Call>())
		return run(expression.get<Call>());

	if (expression.isA<ReferenceAccess>())
		return foldReference(expression);

	if (expression.isA<Operation>())
	{
		auto newexp = foldExpression(expression);

		if (!newexp)
			return newexp.takeError();

		expression = move(*newexp);
		assert(
				expression.isA<Call>() or
						expression.getType().get<BuiltInType>() != Type::unknown().get<BuiltInType>());
		return Error::success();
	}

	assert(false && "unreachable");
	return make_error<NotImplemented>("found a not handled type of expression");
}

std::unique_ptr<Pass> modelica::frontend::createConstantFolderPass()
{
	return std::make_unique<ConstantFolder>();
}
