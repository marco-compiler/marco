#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/SolveDer.h>

using namespace modelica::codegen::model;

DerSolver::DerSolver(SimulationOp simulation, Model& model)
		: simulation(simulation), model(model)
{
}

void DerSolver::solve()
{
	for (auto& equation : model.getEquations())
		solve(*equation);
}

void DerSolver::solve(Equation& equation)
{
	solve<Expression>(equation.lhs());
	solve<Expression>(equation.rhs());
}

template<>
void DerSolver::solve<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		return solve<deref>(expression);
	});
}

template<>
void DerSolver::solve<Constant>(Expression& expression)
{
	// Nothing to do here
}

template<>
void DerSolver::solve<Reference>(Expression& expression)
{
	// Nothing to do here
}

template<>
void DerSolver::solve<Operation>(Expression& expression)
{
	auto* op = expression.getOp();

	if (auto derOp = mlir::dyn_cast<DerOp>(op))
	{
		mlir::OpBuilder builder(simulation);
		mlir::Operation* definingOp = derOp.operand().getDefiningOp();
		llvm::SmallVector<mlir::Value, 3> subscriptions;

		while (!mlir::isa<AllocaOp, AllocOp>(definingOp))
		{
			if (auto subscription = mlir::dyn_cast<SubscriptionOp>(definingOp))
			{
				for (auto index : subscription.indexes())
					subscriptions.push_back(index);

				definingOp = subscription.source().getDefiningOp();
			}
			else
				assert(false && "Unexpected operation");
		}

		assert(definingOp->getNumResults() == 1);
		auto& var = model.getVariable(definingOp->getResult(0));
		mlir::Value derVar;

		if (!var.isState())
		{
			if (auto pointerType = var.getReference().getType().dyn_cast<PointerType>())
				derVar = builder.create<AllocaOp>(derOp.getLoc(), pointerType.getElementType(), pointerType.getShape());
			else
			{
				derVar = builder.create<AllocaOp>(derOp.getLoc(), var.getReference().getType());
			}

			model.addVariable(derVar);
			var.setDer(derVar);
		}
		else
		{
			derVar = var.getDer();
		}

		expression = *Expression::reference(derVar);
		builder.setInsertionPoint(derOp);

		if (!subscriptions.empty())
		{
			auto subscriptionOp = builder.create<SubscriptionOp>(derOp->getLoc(), derVar, subscriptions);
			derVar = subscriptionOp.getResult();
			expression = *Expression::operation(subscriptionOp, std::make_shared<Expression>(expression));
		}

		if (auto pointerType = derVar.getType().cast<PointerType>(); pointerType.getRank() == 0)
			derVar = builder.create<LoadOp>(derOp->getLoc(), derVar);

		derOp.replaceAllUsesWith(derVar);
		derOp->erase();
	}
	else
	{
		for (auto& arg : expression.get<Operation>())
			solve<Expression>(*arg);
	}
}
