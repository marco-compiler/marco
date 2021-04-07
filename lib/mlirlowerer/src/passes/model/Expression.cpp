#include <llvm/ADT/SmallVector.h>
#include <modelica/mlirlowerer/passes/model/Constant.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Operation.h>
#include <modelica/mlirlowerer/passes/model/Reference.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

using namespace modelica::codegen::model;

Expression::Expression(mlir::Operation* op, Constant content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Expression(mlir::Operation* op, Reference content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Expression(mlir::Operation* op, Operation content)
		: op(op), content(content), name(op->getName().getStringRef().str())
{
}

Expression::Ptr Expression::build(mlir::Value value)
{
	mlir::Operation* definingOp = value.getDefiningOp();

	if (auto op = mlir::dyn_cast<LoadOp>(definingOp))
		return build(op.memory());

	if ( mlir::isa<AllocaOp>(definingOp))
		return Expression::reference(value);

	if (mlir::isa<AllocOp>(definingOp))
		return Expression::reference(value);

	if (mlir::isa<ConstantOp>(definingOp))
		return Expression::constant(value);

	llvm::SmallVector<Expression::Ptr, 3> args;

	if (auto op = mlir::dyn_cast<CallOp>(definingOp))
	{
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

	assert(false && "Unexpected operation");
}

Expression::Ptr Expression::constant(mlir::Value value)
{
	return std::make_shared<Expression>(value.getDefiningOp(), Constant(value));
}

Expression::Ptr Expression::reference(mlir::Value value)
{
	return std::make_shared<Expression>(value.getDefiningOp(), Reference(value));
}

Expression::Ptr Expression::operation(mlir::Operation* op, llvm::ArrayRef<std::shared_ptr<Expression>> args)
{
	return std::make_shared<Expression>(op, Operation(args));
}

mlir::Operation* Expression::getOp() const
{
	return op;
}

bool Expression::isConstant() const
{
	return std::holds_alternative<Constant>(content);
}

bool Expression::isReference() const
{
	return std::holds_alternative<Reference>(content);
}

bool Expression::isReferenceAccess() const
{
	if (isReference())
		return true;

	if (isOperation())
		if (mlir::isa<SubscriptionOp>(op))
			return get<Operation>()[0]->isReferenceAccess();

	return false;
}

bool Expression::isOperation() const
{
	return std::holds_alternative<Operation>(content);
}

size_t Expression::childrenCount() const
{
	if (isConstant())
		return 0;

	if (isReference())
		return 0;

	return get<Operation>().size();
}

Expression::Ptr Expression::getChild(size_t index) const
{
	assert(index < childrenCount());
	return get<Operation>()[index];
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
