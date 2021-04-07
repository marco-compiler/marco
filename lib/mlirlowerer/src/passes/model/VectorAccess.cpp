#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Reference.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/utils/IndexSet.hpp>
#include <modelica/utils/Interval.hpp>
#include <string>

using namespace modelica::codegen;
using namespace modelica::codegen::model;
using namespace std;
using namespace llvm;

SingleDimensionAccess::SingleDimensionAccess()
		: value(0), inductionVar(std::numeric_limits<size_t>::max()), isAbs(true)
{
}

SingleDimensionAccess::SingleDimensionAccess(int64_t value, bool isAbs, size_t inductionVar)
		: value(value), inductionVar(inductionVar), isAbs(isAbs)
{
}

bool SingleDimensionAccess::operator==(const SingleDimensionAccess& other) const
{
	if (isAbs != other.isAbs)
		return false;

	if (isAbs)
		return value == other.value;

	return value == other.value && inductionVar == other.inductionVar;
}

int64_t SingleDimensionAccess::getOffset() const
{
	return value;
}

size_t SingleDimensionAccess::getInductionVar() const
{
	return inductionVar;
}

bool SingleDimensionAccess::isOffset() const
{
	return !isAbs;
}

bool SingleDimensionAccess::isDirecAccess() const
{
	return isAbs;
}

modelica::Interval SingleDimensionAccess::map(const MultiDimInterval& multiDimInterval) const
{
	if (isDirecAccess())
		return Interval(value, value + 1);

	return map(multiDimInterval.at(inductionVar));
}

modelica::Interval SingleDimensionAccess::map(const Interval& interval) const
{
	if (isOffset())
		return Interval(interval.min() + value, interval.max() + value);

	return Interval(value, value + 1);
}

size_t SingleDimensionAccess::map(llvm::ArrayRef<size_t> interval) const
{
	if (isOffset())
		return interval[inductionVar] + value;

	return value;
}

SingleDimensionAccess SingleDimensionAccess::absolute(int64_t absVal)
{
	return SingleDimensionAccess(absVal, true);
}

SingleDimensionAccess SingleDimensionAccess::relative(int64_t relativeVal, size_t indVar)
{
	return SingleDimensionAccess(relativeVal, false, indVar);
}

bool SingleDimensionAccess::isCanonical(const Expression& expression)
{
	if (!mlir::isa<SubscriptionOp>(expression.getOp()))
		return false;

	auto subscription = mlir::cast<SubscriptionOp>(expression.getOp());

	for (auto index : subscription.indexes())
	{
		if (mlir::isa<ConstantOp, InductionOp>(index.getDefiningOp()))
			continue;

		// Check whether the accessed position is determined by the sum of
		// and induction variable and a constant.
		if (auto addOp = mlir::dyn_cast<AddOp>(index.getDefiningOp()))
		{
			if (mlir::isa<InductionOp>(addOp.lhs().getDefiningOp()) &&
					mlir::isa<ConstantOp>(addOp.rhs().getDefiningOp()))
				continue;
		}

		// Check whether the accessed position is determined by the difference
		// of and induction variable and a constant.
		if (auto subOp = mlir::dyn_cast<SubOp>(index.getDefiningOp()))
		{
			if (mlir::isa<InductionOp>(subOp.lhs().getDefiningOp()) &&
					mlir::isa<ConstantOp>(subOp.rhs().getDefiningOp()))
				continue;
		}

		// All the checks have failed. The index is not canonical.
		return false;
	}

	return true;
}

VectorAccess::VectorAccess(llvm::SmallVector<SingleDimensionAccess, 3> vector)
		: vectorAccess(std::move(vector))
{
}

bool VectorAccess::operator==(const VectorAccess& other) const
{
	return vectorAccess == other.vectorAccess;
}

bool VectorAccess::operator!=(const VectorAccess& other) const
{
	return !(*this == other);
}

VectorAccess VectorAccess::operator*(const VectorAccess& other) const
{
	return combine(other);
}

modelica::IndexSet VectorAccess::operator*(const IndexSet& other) const
{
	return map(other);
}

VectorAccess::iterator VectorAccess::begin()
{
	return vectorAccess.begin();
}

VectorAccess::const_iterator VectorAccess::begin() const
{
	return vectorAccess.begin();
}

VectorAccess::iterator VectorAccess::end()
{
	return vectorAccess.end();
}

VectorAccess::const_iterator VectorAccess::end() const
{
	return vectorAccess.end();
}

const VectorAccess::Container<SingleDimensionAccess>& VectorAccess::getMappingOffset() const
{
	return vectorAccess;
}

bool VectorAccess::isIdentity() const
{
	for (size_t i = 0; i < vectorAccess.size(); ++i)
	{
		if (!vectorAccess[i].isOffset())
			return false;

		if (vectorAccess[i].getInductionVar() != i)
			return false;

		if (vectorAccess[i].getOffset() != 0)
			return false;
	}

	return true;
}

size_t VectorAccess::mappableDimensions() const
{
	return llvm::count_if(
			vectorAccess, [](const auto& acc) { return acc.isOffset(); });
}

modelica::IndexSet VectorAccess::map(const IndexSet& indexSet) const
{
	IndexSet toReturn;

	for (const auto& part : indexSet)
		toReturn.unite(map(part));

	return toReturn;
}

VectorAccess VectorAccess::invert() const
{
	SmallVector<SingleDimensionAccess, 2> intervals;
	intervals.resize(mappableDimensions());

	for (size_t a = 0; a < vectorAccess.size(); a++)
		if (vectorAccess[a].isOffset())
			intervals[vectorAccess[a].getInductionVar()] = SingleDimensionAccess::relative(-vectorAccess[a].getOffset(), a);

	return VectorAccess(move(intervals));
}

VectorAccess VectorAccess::combine(const VectorAccess& other) const
{
	SmallVector<SingleDimensionAccess, 2> intervals;

	for (const auto& singleAccess : other.vectorAccess)
		intervals.push_back(combine(singleAccess));

	return VectorAccess(move(intervals));
}

SingleDimensionAccess VectorAccess::combine(const SingleDimensionAccess& other) const
{
	if (other.isDirecAccess())
		return other;

	assert(other.getInductionVar() <= vectorAccess.size());
	const auto& mapped = vectorAccess[other.getInductionVar()];

	return SingleDimensionAccess::relative(
			mapped.getOffset() + other.getOffset(), mapped.getInductionVar());
}

modelica::MultiDimInterval VectorAccess::map(const MultiDimInterval& interval) const
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

bool VectorAccess::isCanonical(const Expression& expression)
{
	if (expression.isReference())
		return true;

	if (!expression.isReferenceAccess())
		return false;

	mlir::Operation* op = expression.getOp();

	if (!mlir::isa<SubscriptionOp>(op))
		return false;

	if (!SingleDimensionAccess::isCanonical(expression))
		return false;

	return isCanonical(*expression.getChild(0));
}

AccessToVar::AccessToVar(VectorAccess access, mlir::Value var)
		: access(std::move(access)), var(var)
{
}

VectorAccess& AccessToVar::getAccess()
{
	return access;
}

const VectorAccess& AccessToVar::getAccess() const
{
	return access;
}

mlir::Value AccessToVar::getVar() const
{
	return var;
}

static long getIntFromAttribute(mlir::Attribute attribute)
{
	if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>())
		return indexAttr.getInt();

	if (auto booleanAttr = attribute.dyn_cast<BooleanAttribute>())
		return booleanAttr.getValue() ? 1 : 0;

	if (auto integerAttr = attribute.dyn_cast<IntegerAttribute>())
		return integerAttr.getValue();

	if (auto realAttr = attribute.dyn_cast<RealAttribute>())
		return realAttr.getValue();

	assert(false && "Unknown attribute type");
	return 0;
}

AccessToVar AccessToVar::fromExp(const Expression& expression)
{
	assert(VectorAccess::isCanonical(expression));

	if (expression.isReference())
		return AccessToVar(VectorAccess(), expression.get<Reference>().getVar());

	// Until the expression is composed by nested access into each other,
	// build an access displacement for each of them. Then return them and
	// the accessed variable.

	SmallVector<SingleDimensionAccess, 3> access;
	const auto* ptr = &expression;

	while (auto subscription = mlir::dyn_cast<SubscriptionOp>(ptr->getOp()))
	{
		assert(SingleDimensionAccess::isCanonical(*ptr));
		auto indexes = subscription.indexes();

		for (size_t i = indexes.size(); i > 0; --i)
		{
			auto index = indexes[i - 1];

			if (auto constantOp = mlir::dyn_cast<ConstantOp>(index.getDefiningOp()))
			{
				// for x in a:b loop y[K]
				access.push_back(SingleDimensionAccess::absolute(getIntFromAttribute(constantOp.value())));
				continue;
			}

			if (mlir::isa<InductionOp>(index.getDefiningOp()))
			{
				// for i in a:b loop x[i]
				auto forEquation = subscription->getParentOfType<ForEquationOp>();
				long inductionIndex = forEquation.getInductionIndex(index);
				assert(inductionIndex != -1);
				access.push_back(SingleDimensionAccess::relative(0, inductionIndex));
				continue;
			}

			if (auto addOp = mlir::dyn_cast<AddOp>(index.getDefiningOp()))
			{
				// for i in a:b loop x[i + K]
				auto forEquation = subscription->getParentOfType<ForEquationOp>();

				auto lhs = addOp.lhs();
				assert(mlir::isa<InductionOp>(lhs.getDefiningOp()));

				auto rhs = addOp.rhs();
				assert(mlir::isa<ConstantOp>(rhs.getDefiningOp()));

				long inductionIndex = forEquation.getInductionIndex(lhs);
				assert(inductionIndex != -1);

				int64_t offset = getIntFromAttribute(mlir::cast<ConstantOp>(rhs.getDefiningOp()).value());

				access.push_back(SingleDimensionAccess::relative(offset, inductionIndex));
				continue;
			}

			if (auto subOp = mlir::dyn_cast<SubOp>(index.getDefiningOp()))
			{
				// for i in a:b loop x[i - K]
				auto forEquation = subscription->getParentOfType<ForEquationOp>();

				auto lhs = subOp.lhs();
				assert(mlir::isa<InductionOp>(lhs.getDefiningOp()));

				auto rhs = subOp.rhs();
				assert(mlir::isa<ConstantOp>(rhs.getDefiningOp()));

				long inductionIndex = forEquation.getInductionIndex(lhs);
				assert(inductionIndex != -1);

				int64_t offset = getIntFromAttribute(mlir::cast<ConstantOp>(rhs.getDefiningOp()).value());

				access.push_back(SingleDimensionAccess::relative(-1 * offset, inductionIndex));
				continue;
			}

			assert(false && "Invalid access pattern");
		}

		ptr = ptr->getChild(0).get();
	}

	assert(ptr->isReference());
	reverse(begin(access), end(access));
	return AccessToVar(VectorAccess(move(access)), ptr->get<Reference>().getVar());
}
