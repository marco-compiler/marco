#include "modelica/model/ModEquation.hpp"

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
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
	return Error::success();
}

AccessToVar ModEquation::getDeterminedVariable() const
{
	ReferenceMatcher leftHandMatcher;
	leftHandMatcher.visitLeft(*this);
	const auto& fromVariable = leftHandMatcher.at(0);
	assert(VectorAccess::isCanonical(fromVariable.getExp()));
	return AccessToVar::fromExp(fromVariable.getExp());
}

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

	OS << "\n";
}

ModEquation::ModEquation(
		ModExp left,
		ModExp right,
		std::string templateName,
		MultiDimInterval inds,
		bool isForward)
		: body(make_shared<ModEqTemplate>(
					move(left), move(right), move(templateName))),
			inductions(move(inds)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward)
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
