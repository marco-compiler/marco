#include "modelica/omcToModel/OmcToModelPass.hpp"

#include <numeric>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/frontend/Call.hpp"
#include "modelica/frontend/Constant.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/ForEquation.hpp"
#include "modelica/frontend/Member.hpp"
#include "modelica/frontend/ParserErrors.hpp"
#include "modelica/frontend/ReferenceAccess.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/frontend/Type.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModCall.hpp"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/utils/IRange.hpp"
#include "modelica/utils/Interval.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Error OmcToModelPass::lower(Class& cl, const SymbolTable& table)
{
	SymbolTable t(cl, &table);
	for (auto& member : cl.getMembers())
		if (auto error = lower(member, t); error)
			return error;

	for (auto& eq : cl.getEquations())
	{
		auto modEq = lower(eq, t, 0);
		if (!modEq)
			return modEq.takeError();

		model.addEquation(move(*modEq));
	}

	for (auto& eq : cl.getForEquations())
	{
		auto modEq = lower(eq, t);
		if (!modEq)
			return modEq.takeError();

		model.addEquation(move(*modEq));
	}

	return Error::success();
}

static BultinModTypes builtinToBuiltin(BuiltinType type)
{
	switch (type)
	{
		case modelica::BuiltinType::Boolean:
			return BultinModTypes::BOOL;
		case modelica::BuiltinType::Integer:
			return BultinModTypes::INT;
		case modelica::BuiltinType::Float:
			return BultinModTypes::FLOAT;
		case modelica::BuiltinType::String:
		case modelica::BuiltinType::None:
		case modelica::BuiltinType::Unknown:
			assert(false && "not supported");
			return BultinModTypes::BOOL;
	}
	assert(false && "unreachable");
	return BultinModTypes::BOOL;
}

Expected<ModType> OmcToModelPass::lower(
		const Type& tp, const SymbolTable& table)
{
	return ModType(
			builtinToBuiltin(tp.get<BuiltinType>()),
			static_cast<ArrayRef<size_t>>(tp.getDimensions()));
}

static Expected<ModConst> lowerConstant(const Constant& constant)
{
	if (constant.isA<BuiltinType::Float>())
		return ModConst(constant.get<BuiltinType::Float>());
	if (constant.isA<BuiltinType::Integer>())
		return ModConst(constant.get<BuiltinType::Integer>());
	if (constant.isA<BuiltinType::Boolean>())
		return ModConst(constant.get<BuiltinType::Boolean>());

	assert(false && "unreachable");
	return ModConst(0);
}

Expected<ModExp> OmcToModelPass::defaultInitializer(
		const Member& mem, const SymbolTable& table)
{
	const auto& type = mem.getType();

	if (type.isScalar())
	{
		if (type.get<BuiltinType>() == BuiltinType::Boolean)
			return ModExp(ModConst(false));

		if (type.get<BuiltinType>() == BuiltinType::Integer)
			return ModExp(ModConst(0));

		if (type.get<BuiltinType>() == BuiltinType::Float)
			return ModExp(ModConst(0.0F));
	}

	auto tp = lower(type, table);
	if (!tp)
		return tp.takeError();
	return ModExp(ModCall("fill", { ModExp(ModConst(0)) }, *tp));
}

Expected<ModCall> OmcToModelPass::lowerCall(
		Expression& call, const SymbolTable& table)
{
	assert(call.isA<Call>());
	auto& c = call.get<Call>();
	if (!c.getFunction().isA<ReferenceAccess>())
		return make_error<NotImplemented>(
				"only direct function calls are supported");

	const auto& ref = c.getFunction().get<ReferenceAccess>();

	ModCall::ArgsVec args;
	for (size_t i : irange(c.argumentsCount()))
	{
		auto larg = lower(c[i], table);
		if (!larg)
			return larg.takeError();
		args.emplace_back(std::make_unique<ModExp>(move(*larg)));
	}
	auto retType = lower(call.getType(), table);
	if (!retType)
		return retType.takeError();
	return ModCall(ref.getName(), move(args), move(*retType));
}

Expected<ModEquation> OmcToModelPass::lower(
		Equation& eq, const SymbolTable& table, int nestingLevel)
{
	auto left = lower(eq.getLeftHand(), table);
	if (!left)
		return left.takeError();
	ModExp l = move(*left);

	auto right = lower(eq.getRightHand(), table);
	if (!right)
		return left.takeError();
	ModExp r = move(*right);

	if (not eq.getLeftHand().getType().isScalar())
		for (int i : irange(eq.getLeftHand().getType().dimensionsCount()))
		{
			l = ModExp::at(
					move(l), ModExp::induction(ModExp(ModConst(nestingLevel + i))));
			r = ModExp::at(
					move(r), ModExp::induction(ModExp(ModConst(nestingLevel + i))));
		}

	SmallVector<Interval, 3> dimensions;
	if (not eq.getLeftHand().getType().isScalar())
		for (const auto& i : eq.getLeftHand().getType())
			dimensions.emplace_back(0, i);

	return ModEquation(
			move(l),
			move(r),
			"eq_" + to_string(model.getEquations().size()),
			MultiDimInterval(move(dimensions)));
}

Expected<ModEquation> OmcToModelPass::lower(
		ForEquation& eq, const SymbolTable& table)
{
	SymbolTable t(table);
	for (auto& ind : eq.getInductions())
		t.addSymbol(ind);
	auto modEq = lower(eq.getEquation(), t, eq.getInductions().size());
	if (!modEq)
		return modEq;

	SmallVector<Interval, 3> interval;
	for (auto& ind : eq.getInductions())
	{
		if (!ind.getBegin().isA<Constant>() ||
				!ind.getBegin().get<Constant>().isA<BuiltinType::Integer>())
			return make_error<NotImplemented>("induction var must be constant int");

		if (!ind.getEnd().isA<Constant>() ||
				!ind.getEnd().get<Constant>().isA<BuiltinType::Integer>())
			return make_error<NotImplemented>("induction var must be constant int");

		interval.emplace_back(
				ind.getBegin().get<Constant>().get<BuiltinType::Integer>(),
				ind.getEnd().get<Constant>().get<BuiltinType::Integer>() + 1);
	}

	if (not eq.getEquation().getLeftHand().getType().isScalar())
		for (const auto& i : eq.getEquation().getLeftHand().getType())
			interval.emplace_back(0, i);

	modEq->setInductionVars(MultiDimInterval(move(interval)));
	return modEq;
}

static Expected<ModExp> lowerConstant(Expression& c, const SymbolTable table)
{
	assert(c.isA<Constant>());

	if (c.get<Constant>().isA<BuiltinType::Integer>())
		return ModExp(ModConst(c.get<Constant>().as<BuiltinType::Integer>()));

	if (c.get<Constant>().isA<BuiltinType::Float>())
		return ModExp(ModConst(c.get<Constant>().as<BuiltinType::Float>()));

	return make_error<NotImplemented>("unlowerable constant");
}

Expected<ModExp> OmcToModelPass::lowerReference(
		Expression& ref, const SymbolTable& table)
{
	assert(ref.isA<ReferenceAccess>());
	const auto& name = ref.get<ReferenceAccess>().getName();
	assert(table.hasSymbol(name));
	auto& symbol = table[name];
	if (symbol.isA<Member>())
	{
		Expected<ModType> tp = lower(symbol.get<Member>().getType(), table);
		if (!tp)
			return tp.takeError();
		return ModExp(name, move(*tp));
	}

	int induction = symbol.get<Induction>().getInductionIndex();
	if (symbol.isA<Induction>())
		return ModExp::induction(ModExp(ModConst(induction)));
	return make_error<NotImplemented>("unlowerable symbol reference");
}

static Expected<ModExp> lowerNegate(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::negate(move(arguments[0]));
}
static Expected<ModExp> lowerAdd(ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return accumulate(
			arguments.begin() + 1,
			arguments.end(),
			arguments.front(),
			[](ModExp& left, ModExp& right) { return ModExp::add(left, right); });
}
static Expected<ModExp> lowerSubtract(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);

	if (arguments.size() == 1)
	{
		auto exp = tp.getBuiltin() == BultinModTypes::FLOAT
									 ? ModExp(ModConst(-1.0F), tp)
									 : ModExp(ModConst(-1), tp);
		return ModExp::multiply(arguments[0], exp);
	}

	return ModExp::subtract(arguments[0], arguments[1]);
}
static Expected<ModExp> lowerMultiply(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return accumulate(
			arguments.begin() + 1,
			arguments.end(),
			move(arguments.front()),
			[](ModExp& left, ModExp& right) {
				return ModExp::multiply(move(left), move(right));
			});
}
static Expected<ModExp> lowerDivide(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::divide(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerIfelse(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	arguments[1].setType(tp);
	return ModExp::cond(
			move(arguments[0]), move(arguments[1]), move(arguments[2]));
}
static Expected<ModExp> lowerGreater(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::greaterThan(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerGreaterEqual(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::greaterEqual(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerEqual(ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::equal(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerDifferent(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::different(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerLessEqual(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::lessEqual(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerLess(ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::lessThan(move(arguments[0]), move(arguments[1]));
}
static Expected<ModExp> lowerLand(ModType tp, SmallVector<ModExp, 3> arguments)
{
	assert(false && "notimplemented");
	return make_error<NotImplemented>("not implemented");
}
static Expected<ModExp> lowerLor(ModType tp, SmallVector<ModExp, 3> arguments)
{
	assert(false && "notimplemented");
	return make_error<NotImplemented>("not implemented");
}
static Expected<ModExp> lowerSubscription(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	return accumulate(
			arguments.begin() + 1,
			arguments.end(),
			move(arguments.front()),
			[](ModExp& left, ModExp& right) {
				return ModExp::at(move(left), move(right));
			});
}

static Expected<ModExp> lowerMemberLookup(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	assert(false && "notimplemented");
	return make_error<NotImplemented>("not implemented");
}
static Expected<ModExp> lowerPowerOf(
		ModType tp, SmallVector<ModExp, 3> arguments)
{
	for (auto& arg : arguments)
		arg.setType(tp);
	return ModExp::elevate(move(arguments[0]), move(arguments[1]));
}

Expected<ModExp> OmcToModelPass::lowerOperation(
		Expression& op, const SymbolTable& table)
{
	SmallVector<ModExp, 3> arguments;
	for (auto& arg : op.get<Operation>())
	{
		auto newArg = lower(arg, table);
		if (!newArg)
			return newArg.takeError();

		arguments.emplace_back(move(*newArg));
	}
	auto tp = lower(op.getType(), table);
	if (!tp)
		return tp.takeError();

	switch (op.get<Operation>().getKind())
	{
		case OperationKind::negate:
			return lowerNegate(*tp, move(arguments));
		case OperationKind::add:
			return lowerAdd(*tp, move(arguments));
		case OperationKind::subtract:
			return lowerSubtract(*tp, move(arguments));
		case OperationKind::multiply:
			return lowerMultiply(*tp, move(arguments));
		case OperationKind::divide:
			return lowerDivide(*tp, move(arguments));
		case OperationKind::ifelse:
			return lowerIfelse(*tp, move(arguments));
		case OperationKind::greater:
			return lowerGreater(*tp, move(arguments));
		case OperationKind::greaterEqual:
			return lowerGreaterEqual(*tp, move(arguments));
		case OperationKind::equal:
			return lowerEqual(*tp, move(arguments));
		case OperationKind::different:
			return lowerDifferent(*tp, move(arguments));
		case OperationKind::lessEqual:
			return lowerLessEqual(*tp, move(arguments));
		case OperationKind::less:
			return lowerLess(*tp, move(arguments));
		case OperationKind::land:
			return lowerLand(*tp, move(arguments));
		case OperationKind::lor:
			return lowerLor(*tp, move(arguments));
		case OperationKind::subscription:
			return lowerSubscription(*tp, move(arguments));
		case OperationKind::memberLookup:
			return lowerMemberLookup(*tp, move(arguments));
		case OperationKind::powerOf:
			return lowerPowerOf(*tp, move(arguments));
	}

	assert(false && "unrechable");
}

Expected<ModExp> OmcToModelPass::lower(
		Expression& exp, const SymbolTable& table)
{
	if (exp.isA<Constant>())
		return lowerConstant(exp, table);

	if (exp.isA<Call>())
		return lowerCall(exp, table);

	if (exp.isA<ReferenceAccess>())
		return lowerReference(exp, table);

	if (exp.isA<Operation>())
		return lowerOperation(exp, table);

	assert(false && "unreachable");
	return make_error<NotImplemented>("unrechable");
}

Expected<ModExp> OmcToModelPass::lowerStart(
		Member& member, const SymbolTable& table)
{
	if (not member.hasStartOverload())
		return defaultInitializer(member, table);

	if (not member.getStartOverload().isA<Constant>())
		return make_error<NotImplemented>(
				"Start overload of member " + member.getName() +
				" was not folded into a constant");

	auto cst = lowerConstant(member.getStartOverload().get<Constant>());
	if (!cst)
		return cst.takeError();

	if (member.getType().isScalar())
		return ModExp(*cst);

	auto tp = lower(member.getType(), table);
	if (!tp)
		return tp.takeError();

	return ModExp(ModCall("fill", { ModExp(*cst) }, *tp));
}

Expected<ModExp> OmcToModelPass::initializer(
		Member& member, const SymbolTable& table)
{
	if (!member.hasInitializer())
		return lowerStart(member, table);

	return lower(member.getInitializer(), table);
}

Error OmcToModelPass::lower(Member& member, const SymbolTable& table)
{
	auto tp = lower(member.getType(), table);
	if (!tp)
		return tp.takeError();

	auto initExp = initializer(member, table);
	if (!initExp)
		return initExp.takeError();

	model.emplaceVar(
			member.getName(), move(*initExp), false, member.isParameter());
	return Error::success();
}
