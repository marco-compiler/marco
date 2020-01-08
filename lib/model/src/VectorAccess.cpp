#include "modelica/model/VectorAccess.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

VectorAccess VectorAccess::invert() const
{
	SmallVector<Displacement, 2> intervals;
	intervals.resize(mappableDimensions());
	for (size_t a = 0; a < vectorAccess.size(); a++)
		if (vectorAccess[a].isOffset())
			intervals[vectorAccess[a].getInductionVar()] =
					Displacement(-vectorAccess[a].getOffset(), false, a);

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
 * expression must be a at operator, therefore the left hand
 * is expression rappresenting a vector of some kind, while right must be either
 * a ind, a sum/subtraction of ind and a constant, or a single scalar
 */
Optional<Displacement> modelica::toDisplacement(const ModExp& expression)
{
	assert(expression.getKind() == ModExpKind::at);	 // NOLINT
	const auto& index = expression.getRightHand();

	if (index.isConstant<int>())
		return Displacement(index.getConstant<int>().get(0), true, 0);

	// if the accessing expression is a induction we are in the case were there
	// is no induction var.
	if (index.isOperation() && index.getKind() == ModExpKind::induction)
	{
		const auto& inductionVar = index.getLeftHand();
		if (!inductionVar.isConstant<int>())
			return Optional<Displacement>();

		return Displacement(0, false, inductionVar.getConstant<int>().get(0));
	}

	// we are able to handle constant offsets.
	if (!index.isOperation() || (index.getKind() != ModExpKind::add &&
															 index.getKind() != ModExpKind::sub))
		return Optional<Displacement>();

	// they must be in the form induction var + constant
	const auto& inductionExp = index.getLeftHand();
	const auto& constantExp = index.getRightHand();

	// if left and is not a induction var we return
	if (!inductionExp.isOperation() ||
			inductionExp.getKind() != ModExpKind::induction)
		return Optional<Displacement>();

	// if right is not a constant int we return.
	if (!constantExp.isConstant<int>())
		return Optional<Displacement>();

	// the induction var must be constant
	if (!inductionExp.getLeftHand().isConstant<int>())
		return Optional<Displacement>();

	int multiplier = index.getKind() == ModExpKind::sub ? -1 : 1;
	return Displacement(
			constantExp.getConstant<int>().get(0) * multiplier,
			false,
			inductionExp.getLeftHand().getConstant<int>().get(0));
}

Optional<VectorAccess> modelica::toVectorAccess(const ModExp& expression)
{
	assert(expression.getKind() == ModExpKind::at);	 // NOLINT

	// while the expression is composed by nested access into each
	// other build a access displacement by each of them. then return them and
	// the accessed variable.
	SmallVector<Displacement, 3> access;
	auto ptr = &expression;
	while (ptr->isOperation() && ptr->getKind() == ModExpKind::at)
	{
		auto vAccess = toDisplacement(*ptr);
		if (!vAccess)
			return Optional<VectorAccess>();

		access.push_back(move(*vAccess));
		ptr = &ptr->getLeftHand();
	}

	reverse(begin(access), end(access));

	assert(ptr->isReference());	 // NOLINT

	return VectorAccess(ptr->getReference(), move(access));
}
