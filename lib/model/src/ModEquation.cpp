#include "modelica/model/ModEquation.hpp"

#include <llvm/ADT/StringRef.h>
#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "modelica/model/ModConst.hpp"
#include "modelica/model/ModErrors.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModMatchers.hpp"
#include "modelica/model/ModType.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

/***
 * \return true if kind is associative and commutative
 */
static bool isAssComm(ModExpKind kind)
{
	switch (kind)
	{
		case modelica::ModExpKind::zero:
		case modelica::ModExpKind::greaterEqual:
		case modelica::ModExpKind::greaterThan:
		case modelica::ModExpKind::lessEqual:
		case modelica::ModExpKind::less:
		case modelica::ModExpKind::different:
		case modelica::ModExpKind::negate:
		case modelica::ModExpKind::divide:
		case modelica::ModExpKind::sub:
		case modelica::ModExpKind::induction:
		case modelica::ModExpKind::at:
		case modelica::ModExpKind::conditional:
		case modelica::ModExpKind::elevation:
		case modelica::ModExpKind::module:
		case modelica::ModExpKind::equal:
			return false;
		case modelica::ModExpKind::add:
		case modelica::ModExpKind::mult:
			return true;
	}
	return false;
}

static ModExp reorder(
		ModExpKind kind,
		const ModType& returnType,
		ModExp&& nonConstExp,
		ModExp&& constant)
{
	array<ModExp, 3> expressions = { move(nonConstExp.getLeftHand()),
																	 move(nonConstExp.getRightHand()),
																	 move(constant) };

	// put the constants after
	stable_sort(
			begin(expressions), end(expressions), [](const auto& l, const auto& r) {
				return !l.isConstant();
			});

	ModExp inner(kind, returnType, move(expressions[1]), move(expressions[2]));

	return ModExp(kind, returnType, move(expressions[0]), move(inner));
}

static void removeSubtraction(ModExp& exp)
{
	assert(exp.isOperation<ModExpKind::sub>());

	exp = ModExp(
			ModExpKind::add,
			exp.getModType(),
			move(exp.getLeftHand()),
			ModExp::negate(move(exp.getRightHand())));
}

static void foldExp(ModExp& expression)
{
	if (!expression.isOperation() || !expression.isBinary())
		return;

	if (!expression.getLeftHand().isConstant() &&
			!expression.getRightHand().isConstant())
		return;

	if (expression.getLeftHand().isConstant() &&
			expression.getRightHand().isConstant())
		return;

	if (expression.isOperation<ModExpKind::sub>())
		removeSubtraction(expression);

	if (!isAssComm(expression.getKind()))
		return;

	// here expression is a binary operation and
	// either left or right is a constant, but not both.
	//
	//
	// if the operation of the expression is the same as the non constant
	// children, and the operation is commutative and associative we can push
	// the operation constant torward the deeper expressions so that it can be
	// folded there.
	//
	if (expression.getRightHand().isOperation(expression.getKind()))
	{
		expression = reorder(
				expression.getKind(),
				expression.getModType(),
				move(expression.getRightHand()),
				move(expression.getLeftHand()));
		return;
	}

	if (expression.getLeftHand().isOperation(expression.getKind()))
	{
		expression = reorder(
				expression.getKind(),
				expression.getModType(),
				move(expression.getLeftHand()),
				move(expression.getRightHand()));
		return;
	}
}

static void recursiveFold(ModExp& expression)
{
	for (auto& child : expression)
		recursiveFold(child);

	foldExp(expression);

	expression.tryFoldConstant();
}

void ModEquation::foldConstants()
{
	recursiveFold(getLeft());
	recursiveFold(getRight());
}

template<ModExpKind kind>
static Error explicitateOp(ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	return make_error<FailedExplicitation>(toExp, argumentIndex);
}

template<>
Error explicitateOp<ModExpKind::negate>(
		ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	toExp = move(toExp.getChild(argumentIndex));
	toNest = ModExp::negate(move(toNest));
	return Error::success();
}

template<>
Error explicitateOp<ModExpKind::mult>(
		ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	auto& toMove =
			argumentIndex == 1 ? toExp.getLeftHand() : toExp.getRightHand();
	toNest = ModExp::divide(move(toNest), move(toMove));
	toExp = move(toExp.getChild(argumentIndex));
	return Error::success();
}

template<>
Error explicitateOp<ModExpKind::add>(
		ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	auto& toMove =
			argumentIndex == 1 ? toExp.getLeftHand() : toExp.getRightHand();
	toNest = ModExp::subtract(move(toNest), move(toMove));
	toExp = move(toExp.getChild(argumentIndex));
	return Error::success();
}

template<>
Error explicitateOp<ModExpKind::sub>(
		ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	if (argumentIndex == 0)
		toNest = ModExp::add(move(toNest), move(toExp.getChild(1)));
	else
		toNest = ModExp::subtract(move(toExp.getChild(0)), move(toNest));

	toExp = move(toExp.getChild(argumentIndex));

	return Error::success();
}

template<>
Error explicitateOp<ModExpKind::divide>(
		ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	if (argumentIndex == 0)
		toNest = ModExp::multiply(move(toNest), move(toExp.getChild(1)));
	else
		toNest = ModExp::divide(move(toExp.getChild(0)), move(toNest));

	toExp = move(toExp.getChild(argumentIndex));

	return Error::success();
}

static Error explicitateExp(ModExp& toExp, size_t argumentIndex, ModExp& toNest)
{
	switch (toExp.getKind())
	{
		case ModExpKind::negate:
			return explicitateOp<ModExpKind::negate>(toExp, argumentIndex, toNest);
		case ModExpKind::add:
			return explicitateOp<ModExpKind::add>(toExp, argumentIndex, toNest);
		case ModExpKind::sub:
			return explicitateOp<ModExpKind::sub>(toExp, argumentIndex, toNest);
		case ModExpKind::at:
			return explicitateOp<ModExpKind::at>(toExp, argumentIndex, toNest);
		case ModExpKind::conditional:
			return explicitateOp<ModExpKind::conditional>(
					toExp, argumentIndex, toNest);
		case ModExpKind::different:
			return explicitateOp<ModExpKind::different>(toExp, argumentIndex, toNest);
		case ModExpKind::divide:
			return explicitateOp<ModExpKind::divide>(toExp, argumentIndex, toNest);
		case ModExpKind::greaterEqual:
			return explicitateOp<ModExpKind::greaterEqual>(
					toExp, argumentIndex, toNest);
		case ModExpKind::greaterThan:
			return explicitateOp<ModExpKind::greaterThan>(
					toExp, argumentIndex, toNest);
		case ModExpKind::induction:
			return explicitateOp<ModExpKind::induction>(toExp, argumentIndex, toNest);
		case ModExpKind::elevation:
			return explicitateOp<ModExpKind::elevation>(toExp, argumentIndex, toNest);
		case ModExpKind::mult:
			return explicitateOp<ModExpKind::mult>(toExp, argumentIndex, toNest);
		case ModExpKind::zero:
			return explicitateOp<ModExpKind::zero>(toExp, argumentIndex, toNest);
		case ModExpKind::less:
			return explicitateOp<ModExpKind::less>(toExp, argumentIndex, toNest);
		case ModExpKind::lessEqual:
			return explicitateOp<ModExpKind::lessEqual>(toExp, argumentIndex, toNest);
		case ModExpKind::equal:
			return explicitateOp<ModExpKind::equal>(toExp, argumentIndex, toNest);
		case ModExpKind::module:
			return explicitateOp<ModExpKind::module>(toExp, argumentIndex, toNest);
	}

	assert(false && "unreachable");
	return Error::success();
}

Error ModEquation::explicitate(size_t argumentIndex, bool left)
{
	auto& toExplicitate = left ? getLeft() : getRight();
	auto& otherExp = !left ? getLeft() : getRight();

	assert(toExplicitate.isOperation());
	assert(argumentIndex < toExplicitate.childCount());

	return explicitateExp(toExplicitate, argumentIndex, otherExp);
}

Error ModEquation::explicitate(const ModExpPath& path)
{
	for (auto index : path)
	{
		auto error = explicitate(index, path.isOnEquationLeftHand());
		if (error)
			return error;
	}
	if (!path.isOnEquationLeftHand())
		getTemplate()->swapLeftRight();
	return Error::success();
}

void ModEquation::dump() const { dump(outs()); }

void ModEquation::dump(llvm::raw_ostream& OS) const
{
	if (!isForward())
		OS << "backward ";
	if (isForCycle)
	{
		OS << "for ";
		dumpInductions(OS);
	}
	if (getTemplate()->getName().empty())
	{
		getLeft().dump(OS);
		OS << " = ";
		getRight().dump(OS);
	}
	else
	{
		OS << "template ";
		OS << getTemplate()->getName();
	}

	if (isMatched())
	{
		OS << " matched ";
		matchedExpPath->print(OS);
	}

	OS << "\n";
}

ModEquation::ModEquation(
		ModExp left,
		ModExp right,
		std::string templateName,
		MultiDimInterval inds,
		bool isForward,
		optional<EquationPath> path)
		: body(make_shared<ModEqTemplate>(
					move(left), move(right), move(templateName))),
			inductions(move(inds)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward),
			matchedExpPath(std::move(path))
{
	if (!isForCycle)
		inductions = { { 0, 1 } };
}

void ModEquation::setInductionVars(MultiDimInterval inds)
{
	isForCycle = !inds.empty();
	if (isForCycle)
		inductions = std::move(inds);
	else
		inds = { { 0, 1 } };
}

static ModExp singleDimAccToExp(const SingleDimensionAccess& access, ModExp exp)
{
	if (access.isDirecAccess())
		return ModExp::at(
				move(exp), ModExp(ModConst(static_cast<int>(access.getOffset()))));

	auto ind = ModExp::induction(
			ModExp(ModConst(static_cast<int>(access.getInductionVar()))));
	auto sum = move(ind) + ModExp(ModConst(static_cast<int>(access.getOffset())));

	return ModExp::at(move(exp), move(sum));
}

static ModExp accessToExp(const VectorAccess& access, ModExp exp)
{
	for (const auto& singleDimAcc : access)
		exp = singleDimAccToExp(singleDimAcc, move(exp));

	return exp;
}

static void composeAccess(ModExp& exp, const VectorAccess& transformation)
{
	auto access = AccessToVar::fromExp(exp);
	auto combinedAccess = transformation * access.getAccess();

	auto newExps = exp.getReferredVectorAccessExp();

	exp = accessToExp(combinedAccess, move(newExps));
}

ModEquation ModEquation::composeAccess(const VectorAccess& transformation) const
{
	auto toReturn = clone(getTemplate()->getName() + "composed");
	auto inverted = transformation.invert();
	toReturn.setInductionVars(inverted.map(getInductions()));

	ReferenceMatcher matcher(toReturn);
	for (auto& matchedExp : matcher)
	{
		auto& exp = toReturn.reachExp(matchedExp);
		::composeAccess(exp, transformation);
	}

	return toReturn;
}

ModEquation ModEquation::normalized() const
{
	assert(getLeft().isReferenceAccess());
	auto access = AccessToVar::fromExp(getLeft()).getAccess();
	auto invertedAccess = access.invert();

	return composeAccess(invertedAccess);
}

using Mult = SmallVector<pair<ModExp, bool>, 3>;
using SumsOfMult = SmallVector<pair<Mult, bool>, 3>;

static void toMult(const ModExp& exp, Mult& out, bool mul = true)
{
	if (exp.isOperation<ModExpKind::mult>())
	{
		for (auto& c : exp)
			toMult(c, out, mul);
		return;
	}
	if (exp.isOperation<ModExpKind::divide>())
	{
		for (auto& c : exp)
			toMult(c, out, !mul);
		return;
	}

	assert(exp.isReferenceAccess() or exp.isConstant());

	out.emplace_back(make_pair(exp, mul));
}

static void toSumsOfMult(const ModExp& exp, SumsOfMult& out, bool sum = true)
{
	if (exp.isOperation<ModExpKind::add>())
	{
		for (auto& c : exp)
			toSumsOfMult(c, out, sum);
		return;
	}
	if (exp.isOperation<ModExpKind::sub>())
	{
		toSumsOfMult(exp.getLeftHand(), out, sum);
		toSumsOfMult(exp.getRightHand(), out, !sum);
		return;
	}
	if (exp.isOperation<ModExpKind::negate>())
	{
		for (auto& c : exp)
			toSumsOfMult(c, out, !sum);
		return;
	}

	out.emplace_back();
	out.back().second = sum;
	toMult(exp, out.back().first);
}

static bool usesMember(const Mult& exp, llvm::StringRef varName)
{
	const auto& isReferenceToVar = [&](const pair<ModExp, bool>& exp) {
		if (!exp.first.isReferenceAccess())
			return false;
		return exp.first.getReferredVectorAccesss() == varName;
	};

	return llvm::find_if(exp, isReferenceToVar) != exp.end();
}

static void removeOneUseOfVar(Mult& exp, llvm::StringRef varName)
{
	const auto& isReferenceToVar = [&](const pair<ModExp, bool>& exp) {
		if (!exp.first.isReferenceAccess())
			return false;
		return exp.first.getReferredVectorAccesss() == varName;
	};

	exp.erase(remove_if(exp, isReferenceToVar), exp.end());
}

static ModExp multToExp(Mult& mult)
{
	auto exp = move(mult[0].first);
	if (!mult[0].second)
		exp = ModExp(ModConst(1.0)) / move(exp);
	for (auto& e : make_range(mult.begin() + 1, mult.end()))
	{
		if (e.second)
			exp = move(exp) * move(e.first);
		else
			exp = move(exp) / move(e.first);
	}

	return exp;
}

static ModExp multToExp(pair<Mult, bool>& mult)
{
	auto exp = multToExp(mult.first);
	if (mult.second)
		return exp;

	return ModExp(ModConst(-1.0)) * move(exp);
}

static ModExp sumOfMultToExp(ModExp& exp, pair<Mult, bool>& mult)
{
	return move(exp) + multToExp(mult);
}

ModEquation ModEquation::groupLeftHand() const
{
	auto copy = clone(getTemplate()->getName() + "grouped");
	copy.getRight().distribuiteMultiplications();
	auto acc = AccessToVar::fromExp(getLeft());

	SumsOfMult sums;
	toSumsOfMult(copy.getRight(), sums);

	auto pos = partition(sums, [&](const auto& mult) {
		return usesMember(mult.first, acc.getVarName());
	});

	if (pos == sums.begin())
		return *this;

	for (auto& use : make_range(sums.begin(), pos))
	{
		use.second = !use.second;
		removeOneUseOfVar(use.first, acc.getVarName());
	}

	auto rightHand =
			accumulate(pos + 1, sums.end(), multToExp(*pos), sumOfMultToExp);
	auto leftAccumulated =
			accumulate(sums.begin(), pos, ModExp(ModConst(1.0)), sumOfMultToExp);
	rightHand = move(rightHand) / move(leftAccumulated);

	copy.getRight() = std::move(rightHand);
	return copy;
}

void ModEquation::setMatchedExp(EquationPath path)
{
	assert(reachExp(path).isReferenceAccess());
	matchedExpPath = path;
}
