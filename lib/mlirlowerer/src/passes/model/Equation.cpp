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

static void composeAccess(Expression& exp, const VectorAccess& transformation)
{
	AccessToVar access = AccessToVar::fromExp(exp);
	VectorAccess combinedAccess = transformation * access.getAccess();

	assert(mlir::isa<SubscriptionOp>(exp.getOp()));
	SubscriptionOp op = mlir::cast<SubscriptionOp>(exp.getOp());
	mlir::OpBuilder builder(op);
	mlir::Location loc = op->getLoc();
	llvm::SmallVector<mlir::Value, 3> indexes;

	// Compute new indexes of the SubscriptionOp.
	for (const SingleDimensionAccess& singleDimensionAccess : combinedAccess)
	{
		mlir::Value index;

		if (singleDimensionAccess.isDirectAccess())
			index = builder.create<ConstantOp>(loc, builder.getIndexAttr(singleDimensionAccess.getOffset()));
		else
		{
			mlir::Value inductionVar = exp.getOp()->getParentOfType<ForEquationOp>().body()->getArgument(singleDimensionAccess.getInductionVar());
			mlir::Value offset = builder.create<ConstantOp>(loc, builder.getIndexAttr(singleDimensionAccess.getOffset()));
			index = builder.create<AddOp>(loc, builder.getIndexType(), inductionVar, offset);
		}

		indexes.push_back(index);
	}

	// Replace the old SubscriptionOp with a new one using the computed indexes.
	mlir::Value newSubscriptionOp = builder.create<SubscriptionOp>(loc, op.source(), indexes);
	op.replaceAllUsesWith(newSubscriptionOp);
	op->erase();
}

Equation Equation::composeAccess(const VectorAccess& transformation) const
{
	Equation toReturn = clone();
	VectorAccess inverted = transformation.invert();
	toReturn.setInductions(inverted.map(getInductions()));

	ReferenceMatcher matcher(toReturn);

	for (ExpressionPath& matchedExp : matcher)
	{
		Expression exp = toReturn.reachExp(matchedExp);
		::composeAccess(exp, transformation);
	}

	return toReturn;
}

mlir::LogicalResult Equation::normalize()
{
	// Get how the left-hand side variable is currently accessed
	VectorAccess access = AccessToVar::fromExp(getMatchedExp()).getAccess();

	// Apply the transformation to the induction range
	setInductions(access.map(getInductions()));

	VectorAccess invertedAccess = access.invert();
	ReferenceMatcher matcher(*this);

	for (ExpressionPath& matchedExp : matcher)
	{
		Expression exp = reachExp(matchedExp);
		::composeAccess(exp, invertedAccess);
	}

	update();

	return mlir::success();
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

mlir::LogicalResult Equation::explicitate()
{
	// Clone the equation for backup in case of failure of the algorithm
	Equation clonedEquation = clone();
	if (auto status = clonedEquation.explicitate(clonedEquation.getMatchedExpressionPath()); failed(status))
	{
		clonedEquation.getOp()->dropAllDefinedValueUses();
		clonedEquation.getOp()->erase();
		return status;
	}

	clonedEquation.impl->matchedExpPath = EquationPath({}, true);

	// If the explicitation algorithm was not successful, it means that the equation
	// is implicit and cannot be explicitated.
	if (clonedEquation.isImplicit())
	{
		clonedEquation.getOp()->dropAllDefinedValueUses();
		clonedEquation.getOp()->erase();
		return mlir::failure();
	}

	// Substitute the current equation with the explicitated one.
	getOp()->dropAllDefinedValueUses();
	getOp()->erase();
	impl = clonedEquation.impl;
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

bool Equation::containsAtMostOne(mlir::Value variable)
{
	ReferenceMatcher matcher(*this);

	unsigned int count = 0;
	for (ExpressionPath& path : matcher)
		if (path.getExpression().getReferredVectorAccess() == variable)
			count++;

	return count <= 1;
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

static bool isOperandConstant(const mlir::Value operand)
{
	if (operand.isa<mlir::BlockArgument>())
		return false;

	if (ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(operand.getDefiningOp()))
		if (!constantOp.value().getType().isa<mlir::IndexType>())
			return true;

	return false;
}

static double getValue(ConstantOp constantOp)
{
	mlir::Attribute attribute = constantOp.value();

	if (IntegerAttribute integer = attribute.dyn_cast<IntegerAttribute>())
		return integer.getValue();

	if (RealAttribute real = attribute.dyn_cast<RealAttribute>())
		return real.getValue();

	assert(false && "Unreachable");
	return 0.0;
}

void Equation::foldConstants()
{
	mlir::MLIRContext* context = getOp()->getContext();
	mlir::OpBuilder builder(getOp());
	llvm::SmallVector<mlir::Operation*, 3> operations;

	for (mlir::Operation& operation : getOp().body()->getOperations())
		if (!mlir::isa<EquationSidesOp>(operation))
			operations.push_back(&operation);

	// Check if an operation has only constants as operands.
	for (mlir::Operation* operation : operations)
	{
		if (!llvm::all_of(operation->getOperands(), isOperandConstant))
			continue;

		llvm::SmallVector<mlir::Value, 2> operands;
		llvm::SmallVector<double, 2> values;

		for (mlir::Value operand : operation->getOperands())
		{
			operands.push_back(operand);
			values.push_back(getValue(mlir::cast<ConstantOp>(operand.getDefiningOp())));
		}

		// At this point, we have an operation where all the operands are constants.
		// So we can substitute the operation with the correct constant value.
		ConstantOp newOp;
		mlir::Location loc = operation->getLoc();
		builder.setInsertionPoint(operation);

		if (mlir::isa<AddOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, values[0] + values[1]));

		else if (mlir::isa<SubOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, values[0] - values[1]));

		else if (mlir::isa<MulOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, values[0] * values[1]));

		else if (mlir::isa<DivOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, values[0] / values[1]));

		else if (mlir::isa<PowOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, pow(values[0], values[1])));

		else if (mlir::isa<Atan2Op>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, atan2(values[0], values[1])));

		else if (mlir::isa<NegateOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, -values[0]));

		else if (mlir::isa<AbsOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, abs(values[0])));

		else if (mlir::isa<SignOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, (values[0] > 0.0) - (values[0] < 0.0)));

		else if (mlir::isa<SqrtOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, sqrt(values[0])));

		else if (mlir::isa<ExpOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, exp(values[0])));

		else if (mlir::isa<LogOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, log(values[0])));

		else if (mlir::isa<Log10Op>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, log10(values[0])));

		else if (mlir::isa<SinOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, sin(values[0])));

		else if (mlir::isa<CosOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, cos(values[0])));

		else if (mlir::isa<TanOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, tan(values[0])));

		else if (mlir::isa<AsinOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, asin(values[0])));

		else if (mlir::isa<AcosOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, acos(values[0])));

		else if (mlir::isa<AtanOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, atan(values[0])));

		else if (mlir::isa<SinhOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, sinh(values[0])));

		else if (mlir::isa<CoshOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, cosh(values[0])));

		else if (mlir::isa<TanhOp>(operation))
			newOp = builder.create<ConstantOp>(loc, RealAttribute::get(context, tanh(values[0])));

		else
			continue;

		// Replace the old operation with the new constant and erase it.
		operation->replaceAllUsesWith(newOp);
		operation->erase();

		for (mlir::Value operand : operands)
			operand.getDefiningOp()->erase();

		update();
	}
}

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

void Equation::update()
{
	cleanOperation();

	EquationSidesOp terminator = getTerminator();
	impl->left = Expression::build(terminator.lhs()[0]);
	impl->right = Expression::build(terminator.rhs()[0]);
}
