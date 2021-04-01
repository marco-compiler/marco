#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Reference.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>

using namespace modelica::codegen::model;

Expression::Expression(mlir::Operation* op, Constant content)
		: op(op), content(content)
{
}

Expression::Expression(mlir::Operation* op, Reference content)
		: op(op), content(content)
{
}

Expression::Expression(mlir::Operation* op, Operation content)
		: op(op), content(content)
{
}

mlir::Operation* Expression::getOp()
{
	return op;
}

mlir::Operation* Expression::getOp() const
{
	return op;
}

size_t Expression::childrenCount() const
{
	return visit([](const auto& obj) { return obj.childrenCount(); });
}

Expression& Expression::getChild(size_t index)
{
	assert(index < childrenCount());
	return get<Operation>()[index];
}

const Expression& Expression::getChild(size_t index) const
{
	assert(index < childrenCount());
	return get<Operation>()[index];
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

ExpressionPath::ExpressionPath(const Expression& exp, llvm::SmallVector<size_t, 3> path, bool left)
		: path(std::move(path), left), exp(&exp)
{
}

ExpressionPath::ExpressionPath(const Expression& exp, EquationPath path)
		: path(std::move(path)), exp(&exp)
{
}

EquationPath::const_iterator ExpressionPath::begin() const
{
	return path.begin();
}

EquationPath::const_iterator ExpressionPath::end() const
{
	return path.end();
}

size_t ExpressionPath::depth() const
{
	return path.depth();
}

const Expression& ExpressionPath::getExp() const
{
	return *exp;
}

const EquationPath& ExpressionPath::getEqPath() const
{
	return path;
}

bool ExpressionPath::isOnEquationLeftHand() const
{
	return path.isOnEquationLeftHand();
}

Expression& ExpressionPath::reach(Expression& exp) const
{
	return path.reach(exp);
}

const Expression& ExpressionPath::reach(const Expression& exp) const
{
	return path.reach(exp);
}
