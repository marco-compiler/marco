#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/SolveDer.h>

using namespace modelica::codegen::model;

DerSolver::DerSolver(Model& model) : model(&model)
{
}

mlir::LogicalResult DerSolver::solve(mlir::OpBuilder& builder)
{
	for (auto& equation : model->getEquations())
	{
		solve(builder, equation);
		equation = Equation::build(equation.getOp());
	}

	return mlir::success();
}

void DerSolver::solve(mlir::OpBuilder& builder, Equation equation)
{
	solve<Expression>(builder, equation.lhs());
	solve<Expression>(builder, equation.rhs());
}

template<>
void DerSolver::solve<Expression>(mlir::OpBuilder& builder, Expression expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		return solve<deref>(builder, expression);
	});
}

template<>
void DerSolver::solve<Constant>(mlir::OpBuilder& builder, Expression expression)
{
	// Nothing to do here
}

template<>
void DerSolver::solve<Reference>(mlir::OpBuilder& builder, Expression expression)
{
	// Nothing to do here
}

template<>
void DerSolver::solve<Operation>(mlir::OpBuilder& builder, Expression expression)
{
	mlir::OpBuilder::InsertionGuard guard(builder);
	auto* op = expression.getOp();

	if (auto derOp = mlir::dyn_cast<DerOp>(op))
	{
		mlir::Value operand = derOp.operand();
		llvm::SmallVector<mlir::Value, 3> subscriptions;

		while (!operand.isa<mlir::BlockArgument>())
		{
			mlir::Operation* definingOp = operand.getDefiningOp();

			if (auto subscription = mlir::dyn_cast<SubscriptionOp>(definingOp))
			{
				for (auto index : subscription.indexes())
					subscriptions.push_back(index);

				operand = subscription.source();
			}
			else
				assert(false && "Unexpected operation");
		}

		auto simulation = op->getParentOfType<SimulationOp>();
		mlir::Value var = simulation.getVariableAllocation(operand);
		auto& variable = model->getVariable(var);
		mlir::Value derVar;

		if (!variable.isState())
		{
			auto terminator = mlir::cast<YieldOp>(var.getParentBlock()->getTerminator());
			builder.setInsertionPointAfter(terminator);

			llvm::SmallVector<mlir::Value, 3> args;

			for (mlir::Value arg : terminator.args())
				args.push_back(arg);

			if (auto pointerType = variable.getReference().getType().dyn_cast<PointerType>())
				derVar = builder.create<AllocOp>(derOp.getLoc(), pointerType.getElementType(), pointerType.getShape(), llvm::None, false);
			else
			{
				derVar = builder.create<AllocOp>(derOp.getLoc(), variable.getReference().getType(), llvm::None, llvm::None, false);
			}

			model->addVariable(derVar);
			variable.setDer(derVar);

			args.push_back(derVar);
			builder.create<YieldOp>(terminator.getLoc(), args);
			terminator.erase();

			auto newArgumentType = derVar.getType().cast<PointerType>().toUnknownAllocationScope();
			simulation.body().addArgument(newArgumentType);
			simulation.print().addArgument(newArgumentType);
		}
		else
		{
			derVar = variable.getDer();
		}

		expression = Expression::reference(derVar);
		builder.setInsertionPoint(derOp);

		// Get argument index
		for (auto [declaration, arg] : llvm::zip(
						 mlir::cast<YieldOp>(simulation.init().front().getTerminator()).args(),
						 simulation.body().getArguments()))
			if (declaration == derVar)
				derVar = arg;

		if (!subscriptions.empty())
		{
			auto subscriptionOp = builder.create<SubscriptionOp>(derOp->getLoc(), derVar, subscriptions);
			derVar = subscriptionOp.getResult();
			expression = Expression::operation(subscriptionOp, expression);
		}

		if (auto pointerType = derVar.getType().cast<PointerType>(); pointerType.getRank() == 0)
			derVar = builder.create<LoadOp>(derOp->getLoc(), derVar);

		derOp.replaceAllUsesWith(derVar);
		derOp->erase();
	}
	else
	{
		for (auto& arg : expression.get<Operation>())
			solve<Expression>(builder, *arg);
	}
}
