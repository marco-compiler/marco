#include <llvm/ADT/SmallVector.h>
#include <marco/mlirlowerer/passes/model/Constant.h>
#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Induction.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Operation.h>
#include <marco/mlirlowerer/passes/model/Reference.h>
#include <marco/mlirlowerer/passes/model/VectorAccess.h>

using namespace marco::codegen::model;
using namespace marco::codegen::modelica;

Expression::Impl::Impl(mlir::Operation* op, Constant content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Impl::Impl(mlir::Operation* op, Reference content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Impl::Impl(mlir::Operation* op, Operation content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Impl::Impl(mlir::Operation* op, Induction content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Expression(mlir::Operation* op, Constant content)
		: impl(std::make_shared<Impl>(op, content))
{
}

Expression::Expression(mlir::Operation* op, Reference content)
		: impl(std::make_shared<Impl>(op, content))
{
}

Expression::Expression(mlir::Operation* op, Operation content)
		: impl(std::make_shared<Impl>(op, content))
{
}

Expression::Expression(mlir::Operation* op, Induction content)
		: impl(std::make_shared<Impl>(op, content))
{
}

bool Expression::operator==(const Expression& rhs) const
{
	return impl == rhs.impl;
}

bool Expression::operator!=(const Expression& rhs) const
{
	return !(rhs == *this);
}

Expression Expression::build(mlir::Value value)
{
	mlir::Operation* definingOp = value.getDefiningOp();

	if (auto arg = value.dyn_cast<mlir::BlockArgument>())
	{
		if (mlir::isa<ForEquationOp>(arg.getOwner()->getParentOp()))
			return Expression::induction(arg);

		assert(mlir::isa<SimulationOp>(arg.getOwner()->getParentOp()));
		return Expression::reference(value.getParentRegion()->getParentOfType<SimulationOp>().getVariableAllocation(value));
	}

	if (auto op = mlir::dyn_cast<LoadOp>(definingOp))
		return build(op.memory());

	if (mlir::isa<ConstantOp>(definingOp))
		return Expression::constant(value);

	if (auto op = mlir::dyn_cast<CallOp>(definingOp))
	{
		llvm::SmallVector<Expression, 3> args;

		for (auto arg : op.args())
			args.push_back(build(arg));

		return Expression::operation(op, args);
	}

	if (auto op = mlir::dyn_cast<SubscriptionOp>(definingOp))
		return Expression::operation(op, build(op.source()));

	if (auto op = mlir::dyn_cast<DerOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<NegateOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<AddOp>(definingOp))
		return Expression::operation(op, build(op.lhs()), build(op.rhs()));

	if (auto op = mlir::dyn_cast<SubOp>(definingOp))
		return Expression::operation(op, build(op.lhs()), build(op.rhs()));

	if (auto op = mlir::dyn_cast<MulOp>(definingOp))
		return Expression::operation(op, build(op.lhs()), build(op.rhs()));

	if (auto op = mlir::dyn_cast<DivOp>(definingOp))
		return Expression::operation(op, build(op.lhs()), build(op.rhs()));

	if (auto op = mlir::dyn_cast<PowOp>(definingOp))
		return Expression::operation(op, build(op.base()), build(op.exponent()));

	if (auto op = mlir::dyn_cast<AbsOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<SignOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<SqrtOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<SinOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<CosOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<TanOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<AsinOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<AcosOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<AtanOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<Atan2Op>(definingOp))
		return Expression::operation(op, build(op.y()), build(op.x()));

	if (auto op = mlir::dyn_cast<SinhOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<CoshOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<TanhOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<ExpOp>(definingOp))
		return Expression::operation(op, build(op.exponent()));

	if (auto op = mlir::dyn_cast<LogOp>(definingOp))
		return Expression::operation(op, build(op.operand()));

	if (auto op = mlir::dyn_cast<Log10Op>(definingOp))
		return Expression::operation(op, build(op.operand()));

	assert(false && "Unexpected operation");
}

Expression Expression::constant(mlir::Value value)
{
	return Expression(value.getDefiningOp(), Constant(value));
}

Expression Expression::reference(mlir::Value value)
{
	return Expression(value.getDefiningOp(), Reference(value));
}

Expression Expression::operation(mlir::Operation* op, llvm::ArrayRef<Expression> args)
{
	return Expression(op, Operation(args));
}

Expression Expression::induction(mlir::BlockArgument arg)
{
	return Expression(arg.getOwner()->getParentOp(), Induction(arg));
}

mlir::Operation* Expression::getOp() const
{
	return impl->op;
}

bool Expression::isConstant() const
{
	return std::holds_alternative<Constant>(impl->content);
}

bool Expression::isReference() const
{
	return std::holds_alternative<Reference>(impl->content);
}

bool Expression::isReferenceAccess() const
{
	if (isReference())
		return true;

	if (isOperation())
		if (mlir::isa<SubscriptionOp>(impl->op))
			return get<Operation>()[0]->isReferenceAccess();

	return false;
}

bool Expression::isOperation() const
{
	return std::holds_alternative<Operation>(impl->content);
}

bool Expression::isInduction() const
{
	return std::holds_alternative<Induction>(impl->content);
}

size_t Expression::childrenCount() const
{
	if (!isOperation())
		return 0;

	return get<Operation>().size();
}

Expression Expression::getChild(size_t index) const
{
	assert(index < childrenCount());
	return *get<Operation>()[index];
}

mlir::Value Expression::getReferredVectorAccess() const
{
	return getReferredVectorAccessExp().get<Reference>().getVar();
}

Expression& Expression::getReferredVectorAccessExp()
{
	assert(isReferenceAccess());

	if (isReference())
		return *this;

	auto* exp = this;

	while (mlir::isa<SubscriptionOp>(exp->getOp()))
		exp = exp->get<Operation>()[0].get();

	return *exp;
}

const Expression& Expression::getReferredVectorAccessExp() const
{
	assert(isReferenceAccess());

	if (isReference())
		return *this;

	const auto* exp = this;

	while (mlir::isa<SubscriptionOp>(exp->getOp()))
		exp = exp->get<Operation>()[0].get();

	return *exp;
}
