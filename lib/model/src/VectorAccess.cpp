#include "modelica/model/VectorAccess.hpp"

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

	return VectorAccess(referredVar, move(intervals));
}

MultiDimInterval VectorAccess::map(const MultiDimInterval& interval) const
{
	assert(interval.dimensions() >= mappableDimensions());	// NOLINT
	SmallVector<Interval, 2> intervals;

	for (const auto& displacement : vectorAccess)
		intervals.push_back(displacement.map(interval));

	return MultiDimInterval(std::move(intervals));
}

/***
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
	auto indVar = inductionVar.getConstant<int>().get(0);
	return SingleDimensionAccess::relative(0, indVar);
}

/***
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
	int64_t offset = constantExp.getConstant<int>().get(0) * multiplier;
	size_t indVar = inductionExp.getLeftHand().getConstant<int>().get(0);

	return SingleDimensionAccess::relative(offset, indVar);
}
/***
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

/***
 * return true if the expression is a (ind K) where K is a constant integer
 */
static bool isCanonicalSingleInductionAccess(const ModExp& index)
{
	return (
			index.isOperation() && index.getKind() == ModExpKind::induction &&
			index.getLeftHand().isConstant<int>());
}

/***
 *
 * single dimensions access can be built from expression in the form
 * (at V K), (at V (ind K)), and (at (+/- V (ind I) K)) where
 * V is the vector, K a constant and I the index of the induction variable
 *
 * that is either constant access, induction access, or sum of induction +
 * constant access
 *
 */
bool modelica::isCanonicalSingleDimensionAccess(const ModExp& expression)
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

/***
 * expression must be a at operator, therefore the left hand
 * is expression rappresenting a vector of some kind, while right must be either
 * a ind, a sum/subtraction of ind and a constant, or a single scalar
 */
SingleDimensionAccess modelica::toSingleDimensionAccess(
		const ModExp& expression)
{
	assert(isCanonicalSingleDimensionAccess(expression));	 // NOLINT
	const auto& index = expression.getRightHand();

	// if the accessing expression is a constant we are in the case
	// for x in a:b loop y[K]
	if (index.isConstant<int>())
		return SingleDimensionAccess::absolute(index.getConstant<int>().get(0));

	// if the accessing expression is a induction we are in the case
	// for x in a:b loop y[x]
	if (index.isOperation() && index.getKind() == ModExpKind::induction)
		return inductionToSingleDimensionAccess(index);

	// else we are in the case
	// for x in a:b loop y[x + K]
	return operationToSingleDimensionAccess(index);
}

/***
 * a canonical vector access is either a reference
 * or a nested series of (at (at ...) access) operation all of which are
 * canonical single dimensions access. that is are all in the forms
 *  (at e (+/- (ind I) K)) | (at e (ind I)) | (at e K)
 *
 */
bool modelica::isCanonicalVectorAccess(const ModExp& expression)
{
	if (expression.isReference())
		return true;

	if (!expression.isOperation() || expression.getKind() != ModExpKind::at)
		return false;

	if (!isCanonicalSingleDimensionAccess(expression))
		return false;

	return isCanonicalVectorAccess(expression.getLeftHand());
}

VectorAccess modelica::toVectorAccess(const ModExp& expression)
{
	assert(isCanonicalVectorAccess(expression));

	if (expression.isReference())
		return VectorAccess(expression.getReference());

	// while the expression is composed by nested access into each
	// other build a access displacement by each of them. then return them and
	// the accessed variable.
	SmallVector<SingleDimensionAccess, 3> access;
	auto ptr = &expression;
	while (ptr->isOperation() && ptr->getKind() == ModExpKind::at)
	{
		access.push_back(toSingleDimensionAccess(*ptr));
		ptr = &ptr->getLeftHand();
	}

	reverse(begin(access), end(access));
	return VectorAccess(ptr->getReference(), move(access));
}

void VectorAccess::dump(raw_ostream& OS) const
{
	OS << referredVar;
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
