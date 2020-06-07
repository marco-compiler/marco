#include "modelica/model/VectorAccess.hpp"

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModExp.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

VectorAccess VectorAccess::invert() const
{
	SmallVector<SingleDimensionAccess, 2> intervals;
	intervals.resize(mappableDimensions());
	for (size_t a = 0; a < vectorAccess.size(); a++)
		if (vectorAccess[a].isOffset())
			intervals[vectorAccess[a].getInductionVar()] =
					SingleDimensionAccess::relative(-vectorAccess[a].getOffset(), a);

	return VectorAccess(move(intervals));
}

VectorAccess VectorAccess::combine(const VectorAccess& other) const
{
	SmallVector<SingleDimensionAccess, 2> intervals;
	for (const auto& singleAccess : other.vectorAccess)
		intervals.push_back(combine(singleAccess));

	return VectorAccess(move(intervals));
}

SingleDimensionAccess VectorAccess::combine(
		const SingleDimensionAccess& other) const
{
	if (other.isDirecAccess())
		return other;
	assert(other.getInductionVar() <= vectorAccess.size());
	const auto& mapped = vectorAccess[other.getInductionVar()];
	return SingleDimensionAccess::relative(
			mapped.getOffset() + other.getOffset(), mapped.getInductionVar());
}

MultiDimInterval VectorAccess::map(const MultiDimInterval& interval) const
{
	assert(interval.dimensions() >= mappableDimensions());	// NOLINT
	SmallVector<Interval, 2> intervals;

	for (const auto& displacement : vectorAccess)
		intervals.push_back(displacement.map(interval));

	return MultiDimInterval(std::move(intervals));
}

SmallVector<size_t, 3> VectorAccess::map(llvm::ArrayRef<size_t> interval) const
{
	SmallVector<size_t, 3> intervals;

	for (const auto& displacement : vectorAccess)
		intervals.push_back(displacement.map(interval));

	return intervals;
}

/**
 * \brief given a expression containing a ind operation return the
 * Single dimension vector access rappresenting it
 *
 * example
 *		for x in a:b loop y[x] = 1 end loop
 *	would become relative induction access based on the first
 *	induction variable and with 0 constant element
 *
 *	\pre expression is an ind operation and contains a constant.
 */
static SingleDimensionAccess inductionToSingleDimensionAccess(
		const ModExp& expression)
{
	assert(
			expression.isOperation() &&
			expression.getKind() == ModExpKind::induction);
	const auto& inductionVar = expression.getLeftHand();
	assert(inductionVar.isConstant<int>());
	auto indVar = inductionVar.getConstant().get<int>(0);
	return SingleDimensionAccess::relative(0, indVar);
}

/**
 * transform an expression in the form (+/- (ind n) K) to a relative
 * SingleDimensionAccess refering to the induction variable (ind n) and with
 * offset K
 */
static SingleDimensionAccess operationToSingleDimensionAccess(
		const ModExp& expression)
{
	auto expKind = expression.getKind();
	assert(
			expression.isOperation() &&
			(expKind == ModExpKind::add || expKind == ModExpKind::sub));

	// they must be in the form induction var + constant
	const auto& inductionExp = expression.getLeftHand();
	const auto& constantExp = expression.getRightHand();

	assert(inductionExp.isOperation());
	assert(inductionExp.getKind() == ModExpKind::induction);
	assert(inductionExp.getLeftHand().isConstant<int>());
	assert(constantExp.isConstant<int>());

	int multiplier = expression.getKind() == ModExpKind::sub ? -1 : 1;
	int64_t offset = constantExp.getConstant().get<int>(0) * multiplier;
	size_t indVar = inductionExp.getLeftHand().getConstant().get<int>(0);

	return SingleDimensionAccess::relative(offset, indVar);
}
/**
 * return true if the expression is a (+/- (ind I) K) where I, K is are constant
 * integer
 */
static bool isCanonicalSumInductionAccess(const ModExp& expression)
{
	auto kind = expression.getKind();
	bool isSumOrSub = (kind == ModExpKind::sub) || (kind == ModExpKind::add);

	if (!isSumOrSub)
		return false;

	const auto& inductionExp = expression.getLeftHand();
	const auto& constantExp = expression.getRightHand();

	if (!inductionExp.isOperation())
		return false;

	if (inductionExp.getKind() != ModExpKind::induction)
		return false;
	if (!inductionExp.getLeftHand().isConstant<int>())
		return false;
	if (!constantExp.isConstant<int>())
		return false;
	return true;
}

/**
 * return true if the expression is a (ind K) where K is a constant integer
 */
static bool isCanonicalSingleInductionAccess(const ModExp& index)
{
	return (
			index.isOperation() && index.getKind() == ModExpKind::induction &&
			index.getLeftHand().isConstant<int>());
}

bool SingleDimensionAccess::isCanonical(const ModExp& expression)
{
	if (expression.getKind() != ModExpKind::at)
		return false;

	const auto& index = expression.getRightHand();

	if (index.isConstant<int>())
		return true;

	if (isCanonicalSingleInductionAccess(index))
		return true;

	return isCanonicalSumInductionAccess(index);
}

SingleDimensionAccess SingleDimensionAccess::fromExp(const ModExp& expression)
{
	assert(isCanonical(expression));	// NOLINT
	const auto& index = expression.getRightHand();

	// if the accessing expression is a constant we are in the case
	// for x in a:b loop y[K]
	if (index.isConstant<int>())
		return SingleDimensionAccess::absolute(index.getConstant().get<int>(0));

	// if the accessing expression is a induction we are in the case
	// for x in a:b loop y[x]
	if (index.isOperation() && index.getKind() == ModExpKind::induction)
		return inductionToSingleDimensionAccess(index);

	// else we are in the case
	// for x in a:b loop y[x + K]
	return operationToSingleDimensionAccess(index);
}

bool VectorAccess::isCanonical(const ModExp& expression)
{
	if (expression.isReference())
		return true;

	if (!expression.isOperation() || expression.getKind() != ModExpKind::at)
		return false;

	if (!SingleDimensionAccess::isCanonical(expression))
		return false;

	return isCanonical(expression.getLeftHand());
}

AccessToVar AccessToVar::fromExp(const ModExp& expression)
{
	assert(VectorAccess::isCanonical(expression));

	if (expression.isReference())
		return AccessToVar(VectorAccess(), expression.getReference());

	// while the expression is composed by nested access into each
	// other build a access displacement by each of them. then return them and
	// the accessed variable.
	SmallVector<SingleDimensionAccess, 3> access;
	auto ptr = &expression;
	while (ptr->isOperation() && ptr->getKind() == ModExpKind::at)
	{
		access.push_back(SingleDimensionAccess::fromExp(*ptr));
		ptr = &ptr->getLeftHand();
	}

	assert(ptr->isReference());
	reverse(begin(access), end(access));
	return AccessToVar(VectorAccess(move(access)), ptr->getReference());
}

void VectorAccess::dump(raw_ostream& OS) const
{
	for (const auto& acc : vectorAccess)
		acc.dump(OS);
}

void SingleDimensionAccess::dump(raw_ostream& OS) const
{
	OS << "[";

	if (isAbs)
		OS << value;
	else
		OS << "I" << inductionVar << " + " << value;

	OS << "]";
}

std::string VectorAccess::toString() const
{
	string toReturn;
	llvm::raw_string_ostream stream(toReturn);
	dump(stream);
	stream.flush();
	return toReturn;
}

bool VectorAccess::isIdentity() const
{
	for (size_t a = 0; a < vectorAccess.size(); a++)
	{
		if (!vectorAccess[a].isOffset())
			return false;

		if (vectorAccess[a].getInductionVar() != a)
			return false;

		if (vectorAccess[a].getOffset() != 0)
			return false;
	}
	return true;
}
