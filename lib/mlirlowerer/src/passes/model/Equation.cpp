#include <mlir/Support/LogicalResult.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen::model;

Equation::Equation(mlir::Operation* op,
									 Expression left,
										 Expression right,
										 std::string templateName,
										 MultiDimInterval inds,
										 bool isForward,
										 std::optional<EquationPath> path)
		: op(op),
			body(std::make_shared<EquationTemplate>(left, right, templateName)),
			inductions(std::move(inds)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward),
			matchedExpPath(std::move(path))
{
	if (!isForCycle)
		inductions = { { 0, 1 } };
}

Equation::Equation(mlir::Operation* op,
									 std::shared_ptr<EquationTemplate> templ,
										 MultiDimInterval interval,
										 bool isForward)
		: op(op),
			body(std::move(templ)),
			inductions(std::move(interval)),
			isForCycle(!inductions.empty()),
			isForwardDirection(isForward)
{
	if (!isForCycle)
		inductions = { { 0, 1 } };
}

mlir::Operation* Equation::getOp() const
{
	return op;
}

Expression& Equation::lhs()
{
	return body->lhs();
}

const Expression& Equation::lhs() const
{
	return body->lhs();
}

Expression& Equation::rhs()
{
	return body->rhs();
}

const Expression& Equation::rhs() const
{
	return body->rhs();
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

size_t Equation::amount() const
{
	if (auto equationOp = mlir::dyn_cast<EquationOp>(op))
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

	if (auto forEquationOp = mlir::dyn_cast<ForEquationOp>(op))
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

std::shared_ptr<EquationTemplate>& Equation::getTemplate()
{
	return body;
}

const std::shared_ptr<EquationTemplate>& Equation::getTemplate() const
{
	return body;
}

const modelica::MultiDimInterval& Equation::getInductions() const
{
	return inductions;
}

void Equation::setInductionVars(MultiDimInterval inds)
{
	isForCycle = !inds.empty();

	if (isForCycle)
		inductions = std::move(inds);
	else
		inductions = { { 0, 1 } };
}

bool Equation::isForEquation() const
{
	return isForCycle;
}

size_t Equation::dimensions() const
{
	return isForCycle ? inductions.dimensions() : 0;
}

bool Equation::isForward() const
{
	return isForwardDirection;
}

void Equation::setForward(bool isForward)
{
	isForwardDirection = isForward;
}

bool Equation::isMatched() const
{
	return matchedExpPath.has_value();
}

Expression& Equation::getMatchedExp()
{
	assert(isMatched());
	return reachExp(matchedExpPath.value());
}

const Expression& Equation::getMatchedExp() const
{
	assert(isMatched());
	return reachExp(matchedExpPath.value());
}

void Equation::setMatchedExp(EquationPath path)
{
	assert(reachExp(path).isReferenceAccess());
	matchedExpPath = path;
}

AccessToVar Equation::getDeterminedVariable() const
{
	assert(isMatched());
	return AccessToVar::fromExp(getMatchedExp());
}

ExpressionPath Equation::getMatchedExpressionPath() const
{
	assert(isMatched());
	return ExpressionPath(getMatchedExp(), *matchedExpPath);
}

namespace modelica::codegen::model
{
	template<typename Op>
	static mlir::LogicalResult explicitate(Expression& toExp, size_t index, Expression& toNest)
	{
		return toExp.getOp()->emitError("Unexpected operation to be explicitated: " + toExp.getOp()->getName().getStringRef());
	}

	template<>
	mlir::LogicalResult explicitate<NegateOp>(Expression& toExp, size_t index, Expression& toNest)
	{
		assert(mlir::isa<NegateOp>(toExp.getOp()));
		auto op = mlir::cast<NegateOp>(toExp.getOp());
		mlir::OpBuilder builder(toNest.getOp()->getBlock()->getTerminator());

		if (index == 0)
		{
			assert(toNest.getOp()->getNumResults() == 1);
			mlir::Operation* right = builder.create<NegateOp>(op->getLoc(), op.operand().getType(), toNest.getOp()->getResult(0));
			toNest = Expression::operation(right, toNest);

			op.getResult().replaceAllUsesWith(op.operand());
			op->erase();
			toExp = *toExp.getChild(index);

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<AddOp>(Expression& toExp, size_t index, Expression& toNest)
	{
		assert(mlir::isa<AddOp>(toExp.getOp()));
		auto op = mlir::cast<AddOp>(toExp.getOp());
		mlir::OpBuilder builder(toNest.getOp()->getBlock()->getTerminator());

		if (index == 0)
		{
			assert(toNest.getOp()->getNumResults() == 1);
			mlir::Operation* right = builder.create<SubOp>(op->getLoc(), op.lhs().getType(), toNest.getOp()->getResult(0), op.rhs());
			toNest = Expression::operation(right, toNest, *toExp.getChild(1));

			op.getResult().replaceAllUsesWith(op.lhs());
			op.erase();
			toExp = *toExp.getChild(index);

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest.getOp()->getNumResults() == 1);
			mlir::Operation* right = builder.create<SubOp>(op->getLoc(), op.rhs().getType(), toNest.getOp()->getResult(0), op.lhs());
			toNest = Expression::operation(right, toNest, *toExp.getChild(0));

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();
			toExp = *toExp.getChild(index);

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	template<>
	mlir::LogicalResult explicitate<SubOp>(Expression& toExp, size_t index, Expression& toNest)
	{
		assert(mlir::isa<SubOp>(toExp.getOp()));
		auto op = mlir::cast<SubOp>(toExp.getOp());
		mlir::OpBuilder builder(toNest.getOp()->getBlock()->getTerminator());

		if (index == 0)
		{
			assert(toNest.getOp()->getNumResults() == 1);
			mlir::Operation* right = builder.create<AddOp>(op->getLoc(), op.lhs().getType(), toNest.getOp()->getResult(0), op.rhs());
			toNest = Expression::operation(right, toNest, *toExp.getChild(1));

			op.getResult().replaceAllUsesWith(op.lhs());
			op->erase();
			toExp = *toExp.getChild(index);

			return mlir::success();
		}

		if (index == 1)
		{
			assert(toNest.getOp()->getNumResults() == 1);
			mlir::Operation* right = builder.create<SubOp>(op->getLoc(), op.rhs().getType(), op.lhs(), toNest.getOp()->getResult(0));
			toNest = Expression::operation(right, toNest, *toExp.getChild(0));

			op.getResult().replaceAllUsesWith(op.rhs());
			op->erase();
			toExp = *toExp.getChild(index);

			return mlir::success();
		}

		return op->emitError("Index out of bounds: " + std::to_string(index));
	}

	static mlir::LogicalResult explicitateExpression(Expression& toExp, size_t index, Expression& toNest)
	{
		mlir::Operation* op = toExp.getOp();

		if (mlir::isa<NegateOp>(op))
			return explicitate<NegateOp>(toExp, index, toNest);

		if (mlir::isa<AddOp>(op))
			return explicitate<AddOp>(toExp, index, toNest);

		if (mlir::isa<SubOp>(op))
			return explicitate<SubOp>(toExp, index, toNest);

		return toExp.getOp()->emitError("Unexpected operation to be explicitated: " + toExp.getOp()->getName().getStringRef());
	}
}


static Expression singleDimAccToExp(const SingleDimensionAccess& access, Expression exp)
{
	modelica::codegen::ModelicaBuilder builder(exp.getOp()->getContext(), 64);
	mlir::Location location = exp.getOp()->getLoc();

	/*
	if (access.isDirecAccess())
	{
		builder.setInsertionPoint(exp.getOp());
		mlir::Value index = builder.create<modelica::codegen::ConstantOp>(location, builder.getIndexAttribute(access.getOffset()));
		mlir::Value source = exp.getOp()->getResult(0);
		auto subscription = builder.create<modelica::codegen::SubscriptionOp>(location, source, index);
	}

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
	auto toReturn = clone(getTemplate()->getName() + "composed");
	auto inverted = transformation.invert();
	toReturn.setInductionVars(inverted.map(getInductions()));

	ReferenceMatcher matcher(toReturn);

	for (auto& matchedExp : matcher)
	{
		auto& exp = toReturn.reachExp(matchedExp);
		::composeAccess(exp, transformation);
	}

	return toReturn;
}

Equation Equation::normalized() const
{
	assert(lhs().isReferenceAccess());
	auto access = AccessToVar::fromExp(lhs()).getAccess();
	auto invertedAccess = access.invert();

	return composeAccess(invertedAccess);
}

Equation Equation::normalizeMatched() const
{
	auto access = AccessToVar::fromExp(getMatchedExp()).getAccess();
	auto invertedAccess = access.invert();

	return composeAccess(invertedAccess);
}

mlir::LogicalResult Equation::explicitate(size_t argumentIndex, bool left)
{
	auto& toExplicitate = left ? lhs() : rhs();
	auto& otherExp = !left ? lhs() : rhs();

	assert(toExplicitate.isOperation());
	assert(argumentIndex < toExplicitate.childrenCount());

	return explicitateExpression(toExplicitate, argumentIndex, otherExp);
}

mlir::LogicalResult Equation::explicitate(const ExpressionPath& path)
{
	for (auto index : path)
	{
		if (auto res = explicitate(index, path.isOnEquationLeftHand()); failed(res))
			return res;

		assert(getOp()->getNumRegions() == 1);
		auto terminator = mlir::cast<EquationSidesOp>(getOp()->getRegion(0).front().getTerminator());
		mlir::OpBuilder builder(terminator);
		builder.create<EquationSidesOp>(terminator->getLoc(), lhs().getOp()->getResults(), rhs().getOp()->getResults());
		terminator->erase();
	}

	if (!path.isOnEquationLeftHand())
	{
		getTemplate()->swapLeftRight();

		assert(getOp()->getNumRegions() == 1);
		auto terminator = mlir::cast<EquationSidesOp>(getOp()->getRegion(0).front().getTerminator());
		mlir::OpBuilder builder(terminator);
		builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
		terminator->erase();
	}

	matchedExpPath = std::nullopt;
	return mlir::success();
}

mlir::LogicalResult Equation::explicitate()
{
	if (auto res = explicitate(getMatchedExpressionPath()); failed(res))
		return res;

	matchedExpPath = EquationPath({}, true);
	return mlir::success();
}

Equation Equation::clone(std::string newName) const
{
	Equation clone = *this;
	clone.body = std::make_shared<EquationTemplate>(*body);
	clone.getTemplate()->setName(std::move(newName));
	return clone;
}

EquationTemplate::EquationTemplate(Expression left, Expression right, std::string name)
		: left(std::make_shared<Expression>(left)),
			right(std::make_shared<Expression>(right)),
			name(std::move(name))
{
}

Expression& EquationTemplate::lhs()
{
	return *left;
}

const Expression& EquationTemplate::lhs() const
{
	return *left;
}

Expression& EquationTemplate::rhs()
{
	return *right;
}

const Expression& EquationTemplate::rhs() const
{
	return *right;
}

std::string& EquationTemplate::getName()
{
	return name;
}

const std::string& EquationTemplate::getName() const
{
	return name;
}

void EquationTemplate::setName(std::string newName)
{
	name = newName;
}

void EquationTemplate::swapLeftRight()
{
	std::swap(lhs(), rhs());
}

EquationPath::EquationPath(llvm::SmallVector<size_t, 3> path, bool left)
		: path(std::move(path)), left(left)
{
}

EquationPath::const_iterator EquationPath::begin() const
{
	return path.begin();
}

EquationPath::const_iterator EquationPath::end() const
{
	return path.end();
}

size_t EquationPath::depth() const
{
	return path.size();
}

bool EquationPath::isOnEquationLeftHand() const
{
	return left;
}

Expression& EquationPath::reach(Expression& exp) const
{
	Expression* e = &exp;

	for (auto i : path)
		e = e->getChild(i).get();

	return *e;
}

const Expression& EquationPath::reach(const Expression& exp) const
{
	const Expression* e = &exp;

	for (auto i : path)
		e = e->getChild(i).get();

	return *e;
}
