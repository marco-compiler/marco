#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Support/LogicalResult.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaBuilder.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>

using namespace marco::codegen;
using namespace marco::codegen::model;
using namespace modelica;

class Equation::Impl
{
	public:
	Impl(mlir::Operation* op,
			 Expression left,
			 Expression right,
			 bool isForward = true,
			 std::optional<EquationPath> path = std::nullopt)
			: op(op),
				left(std::move(left)),
				right(std::move(right)),
				isForwardDirection(isForward),
				matchedExpPath(std::move(path))
	{
	}

	friend class Equation;

	private:
	EquationInterface op;
	Expression left;
	Expression right;
	bool isForwardDirection;
	std::optional<EquationPath> matchedExpPath;
};

Equation::Equation(mlir::Operation* op,
									 Expression left,
									 Expression right,
									 bool isForward,
									 std::optional<EquationPath> path)
		: impl(std::make_shared<Impl>(op, left, right, isForward, path))
{
}

bool Equation::operator==(const Equation& rhs) const
{
	return impl == rhs.impl;
}

bool Equation::operator!=(const Equation& rhs) const
{
	return !(rhs == *this);
}

bool Equation::operator<(const Equation& rhs) const
{
	return impl < rhs.impl;
}

bool Equation::operator>(const Equation& rhs) const
{
	return rhs < *this;
}

bool Equation::operator<=(const Equation& rhs) const
{
	return !(rhs < *this);
}

bool Equation::operator>=(const Equation& rhs) const
{
	return !(*this < rhs);
}

Equation Equation::build(mlir::Operation* op)
{
	if (auto equationOp = mlir::dyn_cast<EquationOp>(op))
		return build(equationOp);

	assert(mlir::isa<ForEquationOp>(op));
	return build(mlir::cast<ForEquationOp>(op));
}

Equation Equation::build(EquationOp op)
{
	auto body = op.body();
	auto terminator = mlir::cast<EquationSidesOp>(body->getTerminator());

	// Left-hand side of the equation
	mlir::ValueRange lhs = terminator.lhs();
	assert(lhs.size() == 1);

	llvm::SmallVector<Expression, 3> lhsExpr;

	for (auto value : lhs)
		lhsExpr.push_back(Expression::build(value));

	// Right-hand side of the equation
	mlir::ValueRange rhs = terminator.rhs();
	assert(rhs.size() == 1);

	llvm::SmallVector<Expression, 3> rhsExpr;

	for (auto value : rhs)
		rhsExpr.push_back(Expression::build(value));

	// Number of values of left-hand side and right-hand side must match
	assert(lhsExpr.size() == rhsExpr.size());

	return Equation(op, lhsExpr[0], rhsExpr[0]);
}

Equation Equation::build(ForEquationOp op)
{
	auto body = op.body();
	auto terminator = mlir::cast<EquationSidesOp>(body->getTerminator());

	// Left-hand side of the equation
	mlir::ValueRange lhs = terminator.lhs();
	assert(lhs.size() == 1);

	llvm::SmallVector<Expression, 3> lhsExpr;

	for (auto value : lhs)
		lhsExpr.push_back(Expression::build(value));

	// Right-hand side of the equation
	mlir::ValueRange rhs = terminator.rhs();
	assert(rhs.size() == 1);

	llvm::SmallVector<Expression, 3> rhsExpr;

	for (auto value : rhs)
		rhsExpr.push_back(Expression::build(value));

	// Number of values of left-hand side and right-hand side must match
	assert(lhsExpr.size() == rhsExpr.size());

	return Equation(op, lhsExpr[0], rhsExpr[0]);
}

EquationInterface Equation::getOp() const
{
	return impl->op;
}

Expression Equation::lhs() const
{
	return impl->left;
}

Expression Equation::rhs() const
{
	return impl->right;
}

void Equation::getEquationsAmount(mlir::ValueRange values, llvm::SmallVectorImpl<long>& amounts) const
{
	for (auto value : values)
	{
		size_t amount = 1;

		if (auto arrayType = value.getType().dyn_cast<ArrayType>())
			amount = arrayType.rawSize();

		amounts.push_back(amount);
	}
}

EquationSidesOp Equation::getTerminator() const
{
	return mlir::cast<EquationSidesOp>(getOp().body()->getTerminator());
}

bool Equation::isForLoop() const
{
	ForEquationOp forEquationOp = mlir::cast<ForEquationOp>(getOp().getOperation());

	for (mlir::BlockArgument arg : forEquationOp.body()->getArguments())
		if (!arg.use_empty())
			return true;

	return false;
}

size_t Equation::amount() const
{
	llvm::SmallVector<long, 3> lhsEquations;
	llvm::SmallVector<long, 3> rhsEquations;

	getEquationsAmount(getOp().lhs(), lhsEquations);
	getEquationsAmount(getOp().rhs(), rhsEquations);

	assert(lhsEquations.size() == rhsEquations.size());
	auto pairs = llvm::zip(lhsEquations, rhsEquations);

	size_t result = 0;

	for (const auto& [l, r] : pairs)
	{
		assert(l != -1 || r != -1);

		if (l == -1)
			result += r;
		else if (r == -1)
			result += l;
		else
		{
			assert(l == r);
			result += l;
		}
	}

	for (size_t i = 0, e = getOp().inductions().size(); i < e; ++i)
	{
		auto forEquationOp = mlir::cast<ForEquationOp>(getOp().getOperation());
		auto inductionOp = forEquationOp.inductionsDefinitions()[i].getDefiningOp<InductionOp>();
		result *= inductionOp.end() + 1 - inductionOp.start();
	}

	return result;
}

marco::MultiDimInterval Equation::getInductions() const
{
	if (!isForLoop())
		return { { 0, 1 } };

	auto forEquationOp = mlir::cast<ForEquationOp>(getOp().getOperation());
	llvm::SmallVector<Interval, 3> intervals;

	for (auto induction : forEquationOp.inductionsDefinitions())
	{
		auto inductionOp = induction.getDefiningOp<InductionOp>();
		intervals.emplace_back(inductionOp.start(), inductionOp.end() + 1);
	}

	return MultiDimInterval(intervals);
}

void Equation::setInductions(MultiDimInterval inductions)
{
	if (inductions.empty())
		inductions = { { 0, 1 } };

	auto forEquationOp = mlir::cast<ForEquationOp>(getOp().getOperation());

	mlir::OpBuilder builder(forEquationOp);
	forEquationOp.inductionsBlock()->clear();
	builder.setInsertionPointToStart(forEquationOp.inductionsBlock());

	llvm::SmallVector<mlir::Value, 3> newInductions;

	for (auto induction : inductions)
		newInductions.push_back(builder.create<InductionOp>(getOp()->getLoc(), induction.min(), induction.max() - 1));

	builder.create<YieldOp>(forEquationOp.getLoc(), newInductions);
}

size_t Equation::dimensions() const
{
	return isForLoop() ? getInductions().dimensions() : 0;
}

bool Equation::isForward() const
{
	return impl->isForwardDirection;
}

void Equation::setForward(bool isForward)
{
	impl->isForwardDirection = isForward;
}

bool Equation::isMatched() const
{
	return impl->matchedExpPath.has_value();
}

Expression Equation::getMatchedExp() const
{
	assert(isMatched());
	return reachExp(impl->matchedExpPath.value());
}

void Equation::setMatchedExp(EquationPath path)
{
	assert(reachExp(path).isReferenceAccess());
	impl->matchedExpPath = path;
}

AccessToVar Equation::getDeterminedVariable() const
{
	assert(isMatched());
	return AccessToVar::fromExp(getMatchedExp());
}

ExpressionPath Equation::getMatchedExpressionPath() const
{
	assert(isMatched());
	return ExpressionPath(getMatchedExp(), *impl->matchedExpPath);
}

/**
 * Transform the equation such that the left hand side, which must correspond to
 * the determined variables, has only induction variables as indexes.
 */
void Equation::normalize()
{
	// Get how the left-hand side variable is currently accessed
	assert(lhs() == getMatchedExp());
	VectorAccess access = AccessToVar::fromExp(getMatchedExp()).getAccess();

	// Apply the transformation to the induction range
	MultiDimInterval newInductions(getInductions());
	for (auto& acc : access)
		if (acc.isOffset())
			newInductions[acc.getInductionVar()] = {
				newInductions[acc.getInductionVar()].min() + acc.getOffset(),
				newInductions[acc.getInductionVar()].max() + acc.getOffset()
			};

	setInductions(newInductions);

	// Create a clone of the equation with an empty body.
	Equation clonedEquation = clone();
	EquationInterface clonedOp = clonedEquation.getOp();
	clonedOp.body()->clear();

	mlir::OpBuilder builder(clonedOp);
	mlir::Location loc = clonedOp->getLoc();
	mlir::BlockAndValueMapping mapper;
	builder.setInsertionPointToStart(clonedOp.body());

	// Map the old induction values with the normalized ones.
	for (size_t i : marco::irange(access.size()))
	{
		if (access[i].isOffset())
		{
			mlir::Value offset = builder.create<ConstantOp>(
					loc, IntegerAttribute::get(builder.getContext(), -access[i].getOffset()));
			mlir::Value newInduction = builder.create<AddOp>(
					loc, IntegerType::get(builder.getContext()), clonedOp.induction(access[i].getInductionVar()), offset);
			mapper.map(getOp().induction(access[i].getInductionVar()), newInduction);
		}
	}

	// Copy all the operations into the cloned equation, by using the new mapped induction values.
	for (mlir::Operation& op : getOp().body()->getOperations())
		builder.clone(op, mapper);

	// Replace the current equation with the normalized equation.
	erase();
	impl->op = clonedEquation.impl->op;
	restoreCanonicity();
	update();
}

mlir::LogicalResult Equation::explicitate(mlir::OpBuilder& builder, size_t argumentIndex, bool left)
{
	EquationSidesOp terminator = getTerminator();
	assert(terminator.lhs().size() == 1);
	assert(terminator.rhs().size() == 1);

	mlir::Value toExplicitate = left ? terminator.lhs()[0] : terminator.rhs()[0];
	mlir::Value otherExp = !left ? terminator.lhs()[0] : terminator.rhs()[0];

	mlir::Operation* op = toExplicitate.getDefiningOp();

	// If the operation is not invertible, return an error
	if (!op->hasTrait<InvertibleOpInterface::Trait>())
		return mlir::failure();

	return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
}

mlir::LogicalResult Equation::explicitate(const ExpressionPath& path)
{
	EquationSidesOp terminator = getTerminator();
	mlir::OpBuilder builder(terminator);

	for (size_t index : path)
	{
		if (auto status = explicitate(builder, index, path.isOnEquationLeftHand()); failed(status))
			return status;
	}

	update();

	if (!path.isOnEquationLeftHand())
	{
		std::swap(impl->left, impl->right);

		builder.setInsertionPointAfter(terminator);
		builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
		terminator->erase();
	}

	impl->matchedExpPath = std::nullopt;
	return mlir::success();
}

/**
 * Explicitate the equation by moving to the left hand side the determined
 * variable and the rest to the right hand side. It will fail if the equation is
 * implicit.
 */
mlir::LogicalResult Equation::explicitate()
{
	// Clone the equation for backup in case of failure of the algorithm
	Equation clonedEquation = clone();
	if (auto status = clonedEquation.explicitate(clonedEquation.getMatchedExpressionPath()); failed(status))
	{
		clonedEquation.erase();
		return status;
	}

	clonedEquation.impl->matchedExpPath = EquationPath({}, true);

	// If the explicitation algorithm was not successful, it means that the equation
	// is implicit and cannot be explicitated.
	if (clonedEquation.isImplicit())
	{
		clonedEquation.erase();
		return mlir::failure();
	}

	// Substitute the current equation with the explicitated one.
	erase();
	impl->op = clonedEquation.impl->op;
	impl->matchedExpPath = clonedEquation.impl->matchedExpPath;
	update();

	return mlir::success();
}

bool Equation::isImplicit()
{
	if (!lhs().isReferenceAccess())
		return true;

	ReferenceMatcher matcher;
	matcher.visit(rhs(), false);

	// The equation is implicit only if the accessed variable on the left hand side
	// also appears in the right hand side of the equation.
	for (ExpressionPath& path : matcher)
		if (path.getExpression().getReferredVectorAccess() == getDeterminedVariable().getVar())
			return true;

	return false;
}

Equation Equation::clone() const
{
	mlir::OpBuilder builder(getOp());
	mlir::Operation* newOp = builder.clone(*getOp());
	Equation clone = build(newOp);

	clone.impl->isForwardDirection = impl->isForwardDirection;
	clone.impl->matchedExpPath = impl->matchedExpPath;

	return clone;
}

void Equation::foldConstants()
{
	mlir::OpBuilder builder(getOp());
	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (mlir::Operation& operation : getOp().body()->getOperations())
		operations.push_back(&operation);

	// If an operation has only constants as operands, we can substitute it with
	// the corresponding constant value and erase the old operation.
	for (mlir::Operation* operation : operations)
		if (operation->hasTrait<FoldableOpInterface::Trait>())
			mlir::cast<FoldableOpInterface>(operation).foldConstants(builder);
}

/**
 * Remove opeartions that has no uses inside the MLIR body of the equation.
 */
void Equation::cleanOperation()
{
	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (mlir::Operation& operation : getOp().body()->getOperations())
		if (!mlir::isa<EquationSidesOp>(operation))
			operations.push_back(&operation);

	// If an operation has no uses, erase it.
	for (mlir::Operation* operation : llvm::reverse(operations))
		if (operation->use_empty())
			operation->erase();
}

/**
 * Restore canonicity of array accesses if it was undone by some transformation
 * like normalize() or replaceUses(). For example x[(i+1)+1] must become x[i+2].
 */
void Equation::restoreCanonicity()
{
	foldConstants();

	for (mlir::Operation& op : getOp().body()->getOperations())
	{
		if (!mlir::isa<SubscriptionOp>(op))
			continue;
		SubscriptionOp subOp = mlir::cast<SubscriptionOp>(op);

		for (mlir::Value index : subOp.indexes())
		{
			if (index.isa<mlir::BlockArgument>())
					continue;

			mlir::OpBuilder builder(subOp.getContext());
			builder.setInsertionPoint(subOp);

			if (AddOp outerOp = mlir::dyn_cast<AddOp>(index.getDefiningOp()))
			{
				if (outerOp.lhs().isa<mlir::BlockArgument>() || !mlir::isa<AddOp>(outerOp.lhs().getDefiningOp()))
					continue;
				AddOp innerOp = mlir::cast<AddOp>(outerOp.lhs().getDefiningOp());
				assert(innerOp.lhs().isa<mlir::BlockArgument>());

				mlir::Value offset = builder.create<AddOp>(subOp.getLoc(), outerOp.resultType(), innerOp.rhs(), outerOp.rhs());
				mlir::Value newIndex = builder.create<AddOp>(subOp.getLoc(), outerOp.resultType(), innerOp.lhs(), offset);
				outerOp.getResult().replaceAllUsesWith(newIndex);
			}
			else if (SubOp outerOp = mlir::dyn_cast<SubOp>(index.getDefiningOp()))
			{
				if (outerOp.lhs().isa<mlir::BlockArgument>() || !mlir::isa<AddOp>(outerOp.lhs().getDefiningOp()))
					continue;
				AddOp innerOp = mlir::cast<AddOp>(outerOp.lhs().getDefiningOp());
				assert(innerOp.lhs().isa<mlir::BlockArgument>());

				mlir::Value offset = builder.create<SubOp>(subOp.getLoc(), outerOp.resultType(), innerOp.rhs(), outerOp.rhs());
				mlir::Value newIndex = builder.create<AddOp>(subOp.getLoc(), outerOp.resultType(), innerOp.lhs(), offset);
				outerOp.getResult().replaceAllUsesWith(newIndex);
			}
		}
	}
}

void Equation::update()
{
	foldConstants();
	cleanOperation();

	EquationSidesOp terminator = getTerminator();
	impl->left = Expression::build(terminator.lhs()[0]);
	impl->right = Expression::build(terminator.rhs()[0]);
}

void Equation::erase()
{
	getOp()->dropAllDefinedValueUses();
	getOp()->erase();
}
