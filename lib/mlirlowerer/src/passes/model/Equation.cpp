#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Support/LogicalResult.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen;
using namespace modelica::codegen::model;

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

		if (auto pointerType = value.getType().dyn_cast<PointerType>())
			amount = pointerType.rawSize();

		amounts.push_back(amount);
	}
}

EquationSidesOp Equation::getTerminator() const
{
	return mlir::cast<EquationSidesOp>(getOp().body()->getTerminator());
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

modelica::MultiDimInterval Equation::getInductions() const
{
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
	return getInductions().dimensions();
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

namespace modelica::codegen::model
{
	static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
	{
		if (auto pointerType = operand.getType().dyn_cast<PointerType>(); pointerType && pointerType.getRank() == 0)
			return builder.create<LoadOp>(operand.getLoc(), operand);

		return operand;
	}

	template<typename Op>
	static mlir::LogicalResult explicitate(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		mlir::Operation* op = toExp.getDefiningOp();
		return op->emitError("Unexpected operation to be explicitated: " + op->getName().getStringRef());
	}

	template<>
	mlir::LogicalResult explicitate<CallOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<CallOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<CallOp>();

		auto module = toExp.getDefiningOp()->getParentOfType<mlir::ModuleOp>();
		auto callee = module.lookupSymbol<mlir::FuncOp>(op.callee());

		if (!callee->hasAttr("inverse"))
			return op->emitError("Function " + callee->getName().getStringRef() + " is not invertible");

		auto inverseAnnotation = callee->getAttrOfType<InverseFunctionsAttribute>("inverse");

		if (!inverseAnnotation.isInvertible(index))
			return op->emitError("Function " + callee->getName().getStringRef() + " is not invertible for argument " + std::to_string(index));

		size_t argsSize = op.args().size();
		llvm::SmallVector<mlir::Value, 3> args;

		for (auto arg : inverseAnnotation.getArgumentsIndexes(index))
		{
			if (arg < argsSize)
			{
				args.push_back(op.args()[arg]);
			}
			else
			{
				assert(index == args.size());
				args.push_back(toNest);
			}
		}

		auto invertedCall = builder.create<CallOp>(toExp.getLoc(), inverseAnnotation.getFunction(index), op.args()[index].getType(), args);

		op->getResult(0).replaceAllUsesWith(op.args()[index]);
		op->erase();

		for (auto& use : toNest.getUses())
			if (use.getOwner() != invertedCall)
				use.set(invertedCall.getResult(0));

		return mlir::success();
	}

	template<>
	mlir::LogicalResult explicitate<NegateOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<NegateOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<NegateOp>();

		if (index == 0)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<NegateOp>(op->getLoc(), nestedOperand.getType(), nestedOperand);

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.operand());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<AddOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<AddOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<AddOp>();

		if (index == 0)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.lhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<SubOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<SubOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<SubOp>();

		if (index == 0)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<AddOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op->erase();

			return mlir::success();
		}

		if (index == 1)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), op.lhs(), nestedOperand);

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<MulOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<AddOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<MulOp>();

		if (index == 0)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.lhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<DivOp>(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		assert(mlir::isa<AddOp>(toExp.getDefiningOp()));
		auto op = toExp.getDefiningOp<DivOp>();

		if (index == 0)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<MulOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			mlir::Value nestedOperand = readValue(builder, toNest);
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), op.lhs(), nestedOperand);

			for (auto& use : toNest.getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	static mlir::LogicalResult explicitateExpression(mlir::OpBuilder& builder, mlir::Value toExp, size_t index, mlir::Value toNest)
	{
		if (mlir::isa<CallOp>(toExp.getDefiningOp()))
			return explicitate<CallOp>(builder, toExp, index, toNest);

		if (mlir::isa<NegateOp>(toExp.getDefiningOp()))
			return explicitate<NegateOp>(builder, toExp, index, toNest);

		if (mlir::isa<AddOp>(toExp.getDefiningOp()))
			return explicitate<AddOp>(builder, toExp, index, toNest);

		if (mlir::isa<SubOp>(toExp.getDefiningOp()))
			return explicitate<SubOp>(builder, toExp, index, toNest);

		if (mlir::isa<MulOp>(toExp.getDefiningOp()))
			return explicitate<MulOp>(builder, toExp, index, toNest);

		if (mlir::isa<DivOp>(toExp.getDefiningOp()))
			return explicitate<DivOp>(builder, toExp, index, toNest);

		mlir::Operation* op = toExp.getDefiningOp();
		return op->emitError("Unexpected operation to be explicitated: " + op->getName().getStringRef());
	}
}

static void composeAccess(Expression& exp, const VectorAccess& transformation)
{
	auto access = AccessToVar::fromExp(exp);
	auto combinedAccess = transformation * access.getAccess();

	assert(mlir::isa<SubscriptionOp>(exp.getOp()));
	auto op = mlir::cast<SubscriptionOp>(exp.getOp());
	mlir::OpBuilder builder(op);
	llvm::SmallVector<mlir::Value, 3> indexes;

	for (const auto& singleDimensionAccess : llvm::enumerate(combinedAccess))
	{
		mlir::Value index;

		if (singleDimensionAccess.value().isDirecAccess())
			index = builder.create<modelica::codegen::ConstantOp>(op->getLoc(), builder.getIndexAttr(singleDimensionAccess.value().getOffset()));
		else
		{
			mlir::Value inductionVar = exp.getOp()->getParentOfType<ForEquationOp>().body()->getArgument(singleDimensionAccess.value().getInductionVar());
			mlir::Value offset = builder.create<ConstantOp>(op->getLoc(), builder.getIndexAttr(singleDimensionAccess.value().getOffset()));
			index = builder.create<AddOp>(op->getLoc(), builder.getIndexType(), inductionVar, offset);
		}

		op.indexes()[singleDimensionAccess.index()].replaceAllUsesWith(index);
	}
}

Equation Equation::composeAccess(const VectorAccess& transformation) const
{
	auto toReturn = clone();
	auto inverted = transformation.invert();
	toReturn.setInductions(inverted.map(getInductions()));

	ReferenceMatcher matcher(toReturn);

	for (auto& matchedExp : matcher)
	{
		auto exp = toReturn.reachExp(matchedExp);
		::composeAccess(exp, transformation);
	}

	return toReturn;
}

mlir::LogicalResult Equation::normalize()
{
	// Get how the left-hand side variable is currently accessed
	auto access = AccessToVar::fromExp(getMatchedExp()).getAccess();

	// Apply the transformation to the induction range
	setInductions(access.map(getInductions()));

	auto invertedAccess = access.invert();
	ReferenceMatcher matcher(*this);

	for (auto& matchedExp : matcher)
	{
		auto exp = reachExp(matchedExp);
		::composeAccess(exp, invertedAccess);
	}

	auto terminator = getTerminator();
	impl->left = Expression::build(terminator.lhs()[0]);
	impl->right = Expression::build(terminator.rhs()[0]);

	return mlir::success();
}

mlir::LogicalResult Equation::explicitate(mlir::OpBuilder& builder, size_t argumentIndex, bool left)
{
	auto terminator = getTerminator();
	assert(terminator.lhs().size() == 1);
	assert(terminator.rhs().size() == 1);

	mlir::Value toExplicitate = left ? terminator.lhs()[0] : terminator.rhs()[0];
	mlir::Value otherExp = !left ? terminator.lhs()[0] : terminator.rhs()[0];

	return explicitateExpression(builder, toExplicitate, argumentIndex, otherExp);
}

mlir::LogicalResult Equation::explicitate(const ExpressionPath& path)
{
	auto terminator = getTerminator();
	mlir::OpBuilder builder(terminator);

	for (auto index : path)
	{
		if (auto status = explicitate(builder, index, path.isOnEquationLeftHand()); failed(status))
			return status;
	}

	impl->left = Expression::build(terminator.lhs()[0]);
	impl->right = Expression::build(terminator.rhs()[0]);

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
	if (auto status = explicitate(getMatchedExpressionPath()); failed(status))
		return status;

	impl->matchedExpPath = EquationPath({}, true);
	return mlir::success();
}

Equation Equation::clone() const
{
	mlir::OpBuilder builder(getOp());
	auto* newOp = builder.clone(*getOp());
	Equation clone = build(newOp);

	clone.impl->isForwardDirection = impl->isForwardDirection;
	clone.impl->matchedExpPath = impl->matchedExpPath;

	return clone;
}

void Equation::update()
{
	auto terminator = getTerminator();
	impl->left = Expression::build(terminator.lhs()[0]);
	impl->right = Expression::build(terminator.rhs()[0]);
}
