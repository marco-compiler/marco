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


Equation::Impl::Impl(mlir::Operation* op,
										 Expression left,
										 Expression right,
										 MultiDimInterval inds,
										 bool isForward,
										 std::optional<EquationPath> path)
		: op(op),
			left(std::move(left)),
			right(std::move(right)),
			inductions(std::move(inds)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward),
			matchedExpPath(std::move(path))
{
	if (!isForCycle)
		inductions = { { 0, 1 } };
}

Equation::Equation(mlir::Operation* op,
									 Expression left,
									 Expression right,
									 MultiDimInterval inds,
									 bool isForward,
									 std::optional<EquationPath> path)
		: impl(std::make_shared<Impl>(op, left, right, inds, isForward, path))
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
	auto& body = op.body();
	auto terminator = mlir::cast<EquationSidesOp>(body.front().getTerminator());

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
	auto& body = op.body();
	auto terminator = mlir::cast<EquationSidesOp>(body.front().getTerminator());

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

	llvm::SmallVector<Interval> intervals;

	for (auto induction : op.inductions())
	{
		auto inductionOp = induction.getDefiningOp<InductionOp>();
		intervals.emplace_back(inductionOp.start(), inductionOp.end() + 1);
	}

	return Equation(op, lhsExpr[0], rhsExpr[0], MultiDimInterval(intervals));
}

mlir::Operation* Equation::getOp() const
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

EquationSidesOp Equation::getTerminator()
{
	if (auto equationOp = mlir::dyn_cast<EquationOp>(getOp()))
		return mlir::cast<EquationSidesOp>(equationOp.body().back().getTerminator());

	assert(mlir::isa<ForEquationOp>(getOp()));
	auto forEquationOp = mlir::cast<ForEquationOp>(getOp());
	return mlir::cast<EquationSidesOp>(forEquationOp.body().back().getTerminator());
}

size_t Equation::amount() const
{
	if (auto equationOp = mlir::dyn_cast<EquationOp>(getOp()))
	{
		llvm::SmallVector<long, 3> lhsEquations;
		llvm::SmallVector<long, 3> rhsEquations;

		getEquationsAmount(equationOp.lhs(), lhsEquations);
		getEquationsAmount(equationOp.rhs(), rhsEquations);

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

		return result;
	}

	if (auto forEquationOp = mlir::dyn_cast<ForEquationOp>(getOp()))
	{
		llvm::SmallVector<long, 3> lhsEquations;
		llvm::SmallVector<long, 3> rhsEquations;

		getEquationsAmount(forEquationOp.lhs(), lhsEquations);
		getEquationsAmount(forEquationOp.rhs(), rhsEquations);

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

		for (auto induction : forEquationOp.inductions())
		{
			auto inductionOp = induction.getDefiningOp<InductionOp>();
			result *= inductionOp.end() + 1 - inductionOp.start();
		}

		return result;
	}

	return 0;
}

const modelica::MultiDimInterval& Equation::getInductions() const
{
	return impl->inductions;
}

void Equation::setInductionVars(MultiDimInterval inds)
{
	impl->isForCycle = !inds.empty();

	if (impl->isForCycle)
	{
		impl->inductions = std::move(inds);

		auto forEquationOp = mlir::cast<ForEquationOp>(getOp());

		mlir::OpBuilder builder(forEquationOp);
		forEquationOp.inductionsBlock()->clear();
		builder.setInsertionPointToStart(forEquationOp.inductionsBlock());

		llvm::SmallVector<mlir::Value, 3> newInductions;

		for (auto induction : impl->inductions)
			newInductions.push_back(builder.create<InductionOp>(getOp()->getLoc(), induction.min(), induction.max() - 1));

		builder.create<YieldOp>(forEquationOp.getLoc(), newInductions);
	}
	else
	{
		impl->inductions = { { 0, 1 } };
	}
}

bool Equation::isForEquation() const
{
	return impl->isForCycle;
}

size_t Equation::dimensions() const
{
	return impl->isForCycle ? impl->inductions.dimensions() : 0;
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
	static mlir::LogicalResult explicitate(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		return toExp->emitError("Unexpected operation to be explicitated: " + toExp->getName().getStringRef());
	}

	template<>
	mlir::LogicalResult explicitate<NegateOp>(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		assert(mlir::isa<NegateOp>(toExp));
		auto op = mlir::cast<NegateOp>(toExp);

		if (index == 0)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<NegateOp>(op->getLoc(), nestedOperand.getType(), nestedOperand);

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.operand());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<AddOp>(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		assert(mlir::isa<AddOp>(toExp));
		auto op = mlir::cast<AddOp>(toExp);

		if (index == 0)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.lhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<SubOp>(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		assert(mlir::isa<SubOp>(toExp));
		auto op = mlir::cast<SubOp>(toExp);

		if (index == 0)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<AddOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op->erase();

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<SubOp>(op->getLoc(), nestedOperand.getType(), op.lhs(), nestedOperand);

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<MulOp>(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		assert(mlir::isa<AddOp>(toExp));
		auto op = mlir::cast<AddOp>(toExp);

		if (index == 0)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.lhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<DivOp>(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		assert(mlir::isa<AddOp>(toExp));
		auto op = mlir::cast<AddOp>(toExp);

		if (index == 0)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<MulOp>(op->getLoc(), nestedOperand.getType(), nestedOperand, op.rhs());

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest->hasTrait<mlir::OpTrait::OneResult>());
			mlir::Value nestedOperand = readValue(builder, toNest->getResult(0));
			auto right = builder.create<DivOp>(op->getLoc(), nestedOperand.getType(), op.lhs(), nestedOperand);

			for (auto& use : toNest->getUses())
				if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
					use.set(right.getResult());

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	static mlir::LogicalResult explicitateExpression(mlir::OpBuilder& builder, mlir::Operation* toExp, size_t index, mlir::Operation* toNest)
	{
		if (mlir::isa<NegateOp>(toExp))
			return explicitate<NegateOp>(builder, toExp, index, toNest);

		if (mlir::isa<AddOp>(toExp))
			return explicitate<AddOp>(builder, toExp, index, toNest);

		if (mlir::isa<SubOp>(toExp))
			return explicitate<SubOp>(builder, toExp, index, toNest);

		if (mlir::isa<MulOp>(toExp))
			return explicitate<MulOp>(builder, toExp, index, toNest);

		if (mlir::isa<DivOp>(toExp))
			return explicitate<DivOp>(builder, toExp, index, toNest);

		return toExp->emitError("Unexpected operation to be explicitated: " + toExp->getName().getStringRef());
	}
}

static Expression singleDimAccToExp(const SingleDimensionAccess& access, Expression exp)
{
	mlir::OpBuilder builder(exp.getOp());
	mlir::Location location = exp.getOp()->getLoc();

	if (access.isDirecAccess())
	{
		mlir::Value index = builder.create<modelica::codegen::ConstantOp>(location, builder.getIndexAttr(access.getOffset()));
		mlir::Value source = exp.getOp()->getResult(0);
		auto subscription = builder.create<modelica::codegen::SubscriptionOp>(location, source, index);
		return Expression::build(subscription.getResult());
	}

	/*
	if (access.isDirecAccess())
		return ModExp::at(
				move(exp), ModExp(ModConst(static_cast<int>(access.getOffset()))));

	auto ind = ModExp::induction(
			ModExp(ModConst(static_cast<int>(access.getInductionVar()))));
	auto sum = move(ind) + ModExp(ModConst(static_cast<int>(access.getOffset())));

	return ModExp::at(move(exp), move(sum));
	 */

	return exp;
}

static Expression accessToExp(const VectorAccess& access, Expression exp)
{
	for (const auto& singleDimAcc : access)
		exp = singleDimAccToExp(singleDimAcc, exp);

	return exp;
}

static void composeAccess(Expression& exp, const VectorAccess& transformation)
{
	auto access = AccessToVar::fromExp(exp);
	auto combinedAccess = transformation * access.getAccess();

	auto newExps = exp.getReferredVectorAccessExp();
	exp = accessToExp(combinedAccess, newExps);
}

Equation Equation::composeAccess(const VectorAccess& transformation) const
{
	auto toReturn = Equation(*this);
	auto inverted = transformation.invert();
	toReturn.setInductionVars(inverted.map(getInductions()));

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
	mlir::OpBuilder builder(getTerminator());

	// Get how the left-hand side variable is currently accessed
	auto access = AccessToVar::fromExp(getMatchedExp()).getAccess();

	// Apply the transformation to the induction range
	setInductionVars(access.map(getInductions()));

	ReferenceMatcher matcher(*this);

	/*
	for (auto& matchedExp : matcher)
	{
		auto exp = toReturn.reachExp(matchedExp);
		::composeAccess(exp, transformation);
	}

	return composeAccess(invertedAccess);
	 */

	return mlir::success();
}

mlir::LogicalResult Equation::explicitate(mlir::OpBuilder& builder, size_t argumentIndex, bool left)
{
	auto terminator = getTerminator();
	assert(terminator.lhs().size() == 1);
	assert(terminator.rhs().size() == 1);

	mlir::Value toExplicitate = left ? terminator.lhs()[0] : terminator.rhs()[0];
	mlir::Value otherExp = !left ? terminator.lhs()[0] : terminator.rhs()[0];

	return explicitateExpression(builder, toExplicitate.getDefiningOp(), argumentIndex, otherExp.getDefiningOp());
}

mlir::LogicalResult Equation::explicitate(const ExpressionPath& path)
{
	auto terminator = getTerminator();
	mlir::OpBuilder builder(terminator);

	for (auto index : path)
	{
		if (auto res = explicitate(builder, index, path.isOnEquationLeftHand()); failed(res))
			return res;
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
	if (auto res = explicitate(getMatchedExpressionPath()); failed(res))
		return res;

	impl->matchedExpPath = EquationPath({}, true);
	return mlir::success();
}

Equation Equation::clone() const
{
	mlir::OpBuilder builder(getOp());
	auto* newOp = builder.clone(*getOp());
	Equation clone = build(newOp);

	clone.impl->inductions = impl->inductions;
	clone.impl->isForCycle = impl->isForCycle;
	clone.impl->isForwardDirection = impl->isForwardDirection;
	clone.impl->matchedExpPath = impl->matchedExpPath;

	return clone;
}

using Mult = llvm::SmallVector<std::pair<Expression, bool>, 3>;
using SumsOfMult = llvm::SmallVector<std::pair<Mult, bool>, 3>;

namespace modelica::codegen::model
{
	/**
	 * Replace a subtraction with a sum between the original first operand
	 * and the negation of the second one.
	 *
	 * @param expression subtraction operation
	 */
	static void removeSubtraction(Expression& expression)
	{
		mlir::Operation* op = expression.getOp();
		assert(mlir::isa<SubOp>(op));
		auto subOp = mlir::cast<SubOp>(op);

		mlir::OpBuilder builder(subOp);
		auto negatedRhs = builder.create<NegateOp>(subOp->getLoc(), subOp.rhs().getType(), subOp.rhs());
		auto addOp = builder.create<AddOp>(subOp->getLoc(), subOp.resultType(), subOp.lhs(), negatedRhs);
		op->replaceAllUsesWith(addOp);
		op->erase();

		expression = Expression::operation(
				addOp,
				expression.getChild(0),
				Expression::operation(negatedRhs, expression.getChild(1)));
	}

	static Expression reorder(Expression nonConstExp, Expression constant)
	{
		std::array<Expression, 3> expressions = { nonConstExp.getChild(0), nonConstExp.getChild(1), constant };

		// Put the constants last
		std::stable_sort(
				begin(expressions), end(expressions), [](const auto& l, const auto& r) {
					return !l.isConstant();
				});

		// TODO
		//Expression inner(kind, returnType, expressions[1], expressions[2]);

		//return Expression(kind, returnType, expressions[0], inner);
	}

	static bool isCommutativeAndAssociative(Expression& expression)
	{
		mlir::Operation* op = expression.getOp();

		// TODO: vector multiplication is neither associative nor commutative
		return mlir::isa<AddOp, MulOp>(op);
	}

	static void foldExp(Expression& expression)
	{
		if (!expression.isOperation() || expression.childrenCount() != 2)
			return;

		if (!expression.getChild(0).isConstant() &&
				!expression.getChild(1).isConstant())
			return;

		if (expression.getChild(0).isConstant() &&
				expression.getChild(1).isConstant())
			return;

		if (mlir::isa<SubOp>(expression.getOp()))
			removeSubtraction(expression);

		// Check if the operation is associative and commutative
		if (!isCommutativeAndAssociative(expression))
			return;

		// Here the expression is a binary operation and
		// either left or right operands is a constant, but not both.
		//
		// If the operation is the same as the non constant child, and the
		// operation is commutative and associative, we can push the constant
		// towards the deeper expressions so that it can be
		// folded there.

		if (expression.getChild(1).getOp()->getName() == expression.getOp()->getName())
		{
			expression = reorder(expression.getChild(1), expression.getChild(0));
			return;
		}

		if (expression.getChild(0).getOp()->getName() == expression.getOp()->getName())
		{
			expression = reorder(expression.getChild(0), expression.getChild(1));
			return;
		}
	}

	static void recursiveFold(Expression expression)
	{
		for (size_t i = 0, e = expression.childrenCount(); i < e; ++i)
			recursiveFold(expression.getChild(i));

		foldExp(expression);

		// TODO expression.tryFoldConstant();
	}

	static void toMult(const Expression& exp, Mult& out, bool mul = true)
	{
		if (mlir::isa<MulOp>(exp.getOp()))
		{
			toMult(exp.getChild(0), out, mul);
			toMult(exp.getChild(1), out, mul);
			return;
		}

		if (mlir::isa<DivOp>(exp.getOp()))
		{
			toMult(exp.getChild(0), out, mul);
			toMult(exp.getChild(1), out, !mul);
			return;
		}

		assert(exp.isReferenceAccess() or exp.isConstant());

		out.emplace_back(std::make_pair(exp, mul));
	}

	static void toSumsOfMult(const Expression& exp, SumsOfMult& out, bool sum = true)
	{
		if (mlir::isa<AddOp>(exp.getOp()))
		{
			toSumsOfMult(exp.getChild(0), out, sum);
			toSumsOfMult(exp.getChild(1), out, sum);
			return;
		}

		if (mlir::isa<SubOp>(exp.getOp()))
		{
			toSumsOfMult(exp.getChild(0), out, sum);
			toSumsOfMult(exp.getChild(1), out, !sum);
			return;
		}

		if (mlir::isa<NegateOp>(exp.getOp()))
		{
			toSumsOfMult(exp.getChild(0), out, !sum);
			return;
		}

		out.emplace_back();
		out.back().second = sum;
		toMult(exp, out.back().first);
	}

}

static bool usesMember(const Mult& exp, mlir::Value var)
{
	const auto& isReferenceToVar = [&](const std::pair<Expression, bool>& exp) {
		if (!exp.first.isReferenceAccess())
			return false;

		return exp.first.getReferredVectorAccess() == var;
	};

	return llvm::find_if(exp, isReferenceToVar) != exp.end();
}

Equation Equation::groupLeftHand() const
{
	/*
	auto copy = clone();
	recursiveFold(copy.rhs());
	copy.rhs().distributeMultiplications();
	auto acc = AccessToVar::fromExp(lhs());

	SumsOfMult sums;
	toSumsOfMult(copy.rhs(), sums);

	auto pos = llvm::partition(sums, [&](const auto& mult) {
		return usesMember(mult.first, acc.getVar());
	});

	if (pos == sums.begin())
		return *this;
	 */

	/*
	if (pos == sums.end())
	{
		copy.getRight() = ModExp(ModConst(0), copy.lhs().getModType());
		return copy;
	}

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

	copy.rhs() = std::move(rightHand);
	return copy;
	 */

	return *this;
}
