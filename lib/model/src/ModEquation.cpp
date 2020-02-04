#include "modelica/model/ModEquation.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "modelica/model/ModExp.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

IndexSet ModEquation::toIndexSet() const
{
	SmallVector<Interval, 2> intervals;

	for (const auto& induction : getInductions())
		intervals.emplace_back(induction.begin(), induction.end());

	return IndexSet({ MultiDimInterval(move(intervals)) });
}

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
		const ModExp& original, ModExp&& nonConstExp, ModExp&& constant)
{
	array<ModExp, 3> expressions = { move(nonConstExp.getLeftHand()),
																	 move(nonConstExp.getRightHand()),
																	 move(constant) };

	// put the constants after
	stable_sort(
			begin(expressions), end(expressions), [](const auto& l, const auto& r) {
				return !l.isConstant();
			});

	ModExp inner(
			original.getKind(),
			original.getModType(),
			llvm::make_unique<ModExp>(move(expressions[1])),
			llvm::make_unique<ModExp>(move(expressions[2])));

	return ModExp(
			original.getKind(),
			original.getModType(),
			llvm::make_unique<ModExp>(move(expressions[0])),
			llvm::make_unique<ModExp>(move(inner)));
}

class ConstantFolderVisitor
{
	public:
	void visit(ModExp& expression)
	{
		if (!expression.isOperation() || !expression.isBinary())
			return;

		if (!expression.getLeftHand().isConstant() &&
				!expression.getRightHand().isConstant())
			return;

		if (expression.getLeftHand().isConstant() &&
				expression.getRightHand().isConstant())
			return;

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
					expression,
					move(expression.getRightHand()),
					move(expression.getLeftHand()));
			return;
		}

		if (expression.getLeftHand().isOperation(expression.getKind()))
		{
			expression = reorder(
					expression,
					move(expression.getLeftHand()),
					move(expression.getRightHand()));
			return;
		}
	}

	void afterVisit(ModExp& expression) { expression.tryFoldConstant(); }
};

void ModEquation::foldConstants()
{
	ConstantFolderVisitor visitor;
	visit(getLeft(), visitor);
	visit(getRight(), visitor);
}
