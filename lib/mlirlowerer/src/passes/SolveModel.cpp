#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/matching/SCCDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/SVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/matching/VVarDependencyGraph.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/utils/Interval.hpp>

using namespace modelica;
using namespace codegen;
using namespace model;

struct LoopifyPattern : public mlir::OpRewritePattern<SimulationOp>
{
	using mlir::OpRewritePattern<SimulationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SimulationOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
		rewriter.eraseOp(terminator);

		llvm::SmallVector<mlir::Value, 3> newVars;
		newVars.push_back(terminator.args()[0]);

		unsigned int index = 1;

		std::map<ForEquationOp, mlir::Value> inductions;

		for (auto it = std::next(terminator.args().begin()); it != terminator.args().end(); ++it)
		{
			mlir::Value value = *it;

			if (auto pointerType = value.getType().dyn_cast<PointerType>(); pointerType && pointerType.getRank() == 0)
			{
				{
					mlir::OpBuilder::InsertionGuard guard(rewriter);
					rewriter.setInsertionPoint(value.getDefiningOp());
					mlir::Value newVar = rewriter.create<AllocOp>(value.getLoc(), pointerType.getElementType(), 1, llvm::None, false);
					mlir::Value zeroValue = rewriter.create<ConstantOp>(value.getLoc(), rewriter.getIndexAttr(0));
					mlir::Value subscription = rewriter.create<SubscriptionOp>(value.getLoc(), newVar, zeroValue);

					value.replaceAllUsesWith(subscription);
					rewriter.eraseOp(value.getDefiningOp());
					value = newVar;
				}

				{
					// Fix body argument
					mlir::OpBuilder::InsertionGuard guard(rewriter);

					auto originalArgument = op.body().getArgument(index);
					auto newArgument = op.body().insertArgument(index + 1, value.getType().cast<PointerType>().toUnknownAllocationScope());

					for (auto useIt = originalArgument.use_begin(); useIt != originalArgument.use_end();)
					{
						auto& use = *useIt;
						++useIt;
						rewriter.setInsertionPoint(use.getOwner());

						auto* parentEquation = use.getOwner()->getParentWithTrait<EquationInterface::Trait>();

						if (auto equationOp = mlir::dyn_cast<EquationOp>(parentEquation))
						{
							auto forEquationOp = convertToForEquation(equationOp, rewriter, equationOp.getLoc());
							inductions[forEquationOp] = forEquationOp.body()->getArgument(0);
							parentEquation = forEquationOp;
						}

						auto forEquation = mlir::cast<ForEquationOp>(parentEquation);

						if (inductions.find(forEquation) == inductions.end())
						{
							mlir::Value i = rewriter.create<ConstantOp>(use.get().getLoc(), rewriter.getIndexAttr(0));
							mlir::Value subscription = rewriter.create<SubscriptionOp>(use.get().getLoc(), newArgument, i);
							use.set(subscription);
						}
						else
						{
							mlir::Value i = inductions[forEquation];
							i = rewriter.create<SubOp>(i.getLoc(), i.getType(), i, rewriter.create<ConstantOp>(i.getLoc(), rewriter.getIndexAttr(1)));
							mlir::Value subscription = rewriter.create<SubscriptionOp>(use.get().getLoc(), newArgument, i);
							use.set(subscription);
						}
					}

					op.body().eraseArgument(index);
				}

				{
					// Fix print argument
					auto originalArgument = op.print().getArgument(index);
					auto newArgument = op.print().insertArgument(index + 1, value.getType().cast<PointerType>().toUnknownAllocationScope());
					originalArgument.replaceAllUsesWith(newArgument);
					op.print().eraseArgument(index);
				}
			}

			newVars.push_back(value);
			++index;
		}

		{
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToEnd(&op.init().back());
			rewriter.create<YieldOp>(terminator.getLoc(), newVars);
		}

		// Replace operation with a new one
		auto result = rewriter.replaceOpWithNewOp<SimulationOp>(op, op.startTime(), op.endTime(), op.timeStep(), op.body().getArgumentTypes());
		rewriter.mergeBlocks(&op.init().front(), &result.init().front(), result.init().getArguments());
		rewriter.mergeBlocks(&op.body().front(), &result.body().front(), result.body().getArguments());
		rewriter.mergeBlocks(&op.print().front(), &result.print().front(), result.print().getArguments());

		return mlir::success();
	}

	private:
	ForEquationOp convertToForEquation(EquationOp equation, mlir::PatternRewriter& rewriter, mlir::Location loc) const
	{
		mlir::OpBuilder::InsertionGuard guard(rewriter);
		rewriter.setInsertionPoint(equation);

		auto forEquation = rewriter.create<ForEquationOp>(loc, 1);

		// Inductions
		rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
		mlir::Value induction = rewriter.create<InductionOp>(loc, 1, 1);
		rewriter.create<YieldOp>(loc, induction);

		// Body
		rewriter.mergeBlocks(equation.body(), forEquation.body());

		rewriter.eraseOp(equation);
		return forEquation;
	}
};

struct EquationOpScalarizePattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		assert(sides.lhs().size() == 1 && sides.rhs().size() == 1);

		auto lhs = sides.lhs()[0];
		auto rhs = sides.rhs()[0];

		auto lhsPointerType = lhs.getType().cast<PointerType>();
		auto rhsPointerType = rhs.getType().cast<PointerType>();
		assert(lhsPointerType.getRank() == rhsPointerType.getRank());

		auto forEquation = rewriter.create<ForEquationOp>(location, lhsPointerType.getRank());

		{
			// Inductions
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
			llvm::SmallVector<mlir::Value, 3> inductions;

			for (const auto& [left, right] : llvm::zip(lhsPointerType.getShape(), rhsPointerType.getShape()))
			{
				assert(left != -1 || right != -1);

				if (left != -1 && right != -1)
					assert(left == right);

				long size = std::max(left, right);
				mlir::Value induction = rewriter.create<InductionOp>(location, 1, size);
				inductions.push_back(induction);
			}

			rewriter.create<YieldOp>(location, inductions);
		}

		{
			// Body
			mlir::OpBuilder::InsertionGuard guard(rewriter);

			rewriter.mergeBlocks(op.body(), forEquation.body());
			rewriter.setInsertionPoint(sides);

			llvm::SmallVector<mlir::Value, 1> newLhs;
			llvm::SmallVector<mlir::Value, 1> newRhs;

			for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
			{
				auto leftSubscription = rewriter.create<SubscriptionOp>(location, lhs, forEquation.inductions());
				newLhs.push_back(rewriter.create<LoadOp>(location, leftSubscription));

				auto rightSubscription = rewriter.create<SubscriptionOp>(location, rhs, forEquation.inductions());
				newRhs.push_back(rewriter.create<LoadOp>(location, rightSubscription));
			}

			rewriter.setInsertionPointAfter(sides);
			rewriter.replaceOpWithNewOp<EquationSidesOp>(sides, newLhs, newRhs);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ForEquationOpScalarizePattern : public mlir::OpRewritePattern<ForEquationOp>
{
	using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(ForEquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		assert(sides.lhs().size() == 1 && sides.rhs().size() == 1);

		auto lhs = sides.lhs()[0];
		auto rhs = sides.rhs()[0];

		auto lhsPointerType = lhs.getType().cast<PointerType>();
		auto rhsPointerType = rhs.getType().cast<PointerType>();
		assert(lhsPointerType.getRank() == rhsPointerType.getRank());

		auto forEquation = rewriter.create<ForEquationOp>(location, lhsPointerType.getRank() + op.inductionsDefinitions().size());

		{
			// Inductions
			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
			llvm::SmallVector<mlir::Value, 3> inductions;

			for (const auto& [left, right] : llvm::zip(lhsPointerType.getShape(), rhsPointerType.getShape()))
			{
				assert(left != -1 || right != -1);

				if (left != -1 && right != -1)
					assert(left == right);

				long size = std::max(left, right);
				mlir::Value induction = rewriter.create<InductionOp>(location, 1, size);
				inductions.push_back(induction);
			}

			for (auto induction : op.inductionsDefinitions())
			{
				mlir::Operation* clone = rewriter.clone(*induction.getDefiningOp());
				inductions.push_back(clone->getResult(0));
			}

			rewriter.create<YieldOp>(location, inductions);
		}

		{
			// Body
			mlir::OpBuilder::InsertionGuard guard(rewriter);

			rewriter.mergeBlocks(op.body(), forEquation.body());
			rewriter.setInsertionPoint(sides);

			llvm::SmallVector<mlir::Value, 1> newLhs;
			llvm::SmallVector<mlir::Value, 1> newRhs;

			mlir::ValueRange allInductions = forEquation.inductions();
			mlir::ValueRange newInductions = mlir::ValueRange(allInductions.begin(), allInductions.begin() + lhsPointerType.getRank());

			for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
			{
				auto leftSubscription = rewriter.create<SubscriptionOp>(location, lhs, newInductions);
				newLhs.push_back(rewriter.create<LoadOp>(location, leftSubscription));

				auto rightSubscription = rewriter.create<SubscriptionOp>(location, rhs, newInductions);
				newRhs.push_back(rewriter.create<LoadOp>(location, rightSubscription));
			}

			rewriter.setInsertionPointAfter(sides);
			rewriter.replaceOpWithNewOp<EquationSidesOp>(sides, newLhs, newRhs);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct DerOpPattern : public mlir::OpRewritePattern<DerOp>
{
	DerOpPattern(mlir::MLIRContext* context, Model& model)
			: mlir::OpRewritePattern<DerOp>(context), model(&model)
	{
	}

	mlir::LogicalResult matchAndRewrite(DerOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Value operand = op.operand();
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
		auto variable = model->getVariable(var);
		mlir::Value derVar;

		if (!variable.isState())
		{
			auto terminator = mlir::cast<YieldOp>(var.getParentBlock()->getTerminator());
			rewriter.setInsertionPointAfter(terminator);

			llvm::SmallVector<mlir::Value, 3> args;

			for (mlir::Value arg : terminator.args())
				args.push_back(arg);

			if (auto pointerType = variable.getReference().getType().dyn_cast<PointerType>())
				derVar = rewriter.create<AllocOp>(op.getLoc(), pointerType.getElementType(), pointerType.getShape(), llvm::None, false);
			else
			{
				derVar = rewriter.create<AllocOp>(op.getLoc(), variable.getReference().getType(), llvm::None, llvm::None, false);
			}

			model->addVariable(derVar);
			variable.setDer(derVar);

			args.push_back(derVar);
			rewriter.create<YieldOp>(terminator.getLoc(), args);
			rewriter.eraseOp(terminator);

			auto newArgumentType = derVar.getType().cast<PointerType>().toUnknownAllocationScope();
			simulation.body().addArgument(newArgumentType);
			simulation.print().addArgument(newArgumentType);
		}
		else
		{
			derVar = variable.getDer();
		}

		rewriter.setInsertionPoint(op);

		// Get argument index
		for (auto [declaration, arg] : llvm::zip(
				mlir::cast<YieldOp>(simulation.init().front().getTerminator()).args(),
				simulation.body().getArguments()))
			if (declaration == derVar)
				derVar = arg;

		if (!subscriptions.empty())
		{
			auto subscriptionOp = rewriter.create<SubscriptionOp>(op->getLoc(), derVar, subscriptions);
			derVar = subscriptionOp.getResult();
		}

		if (auto pointerType = derVar.getType().cast<PointerType>(); pointerType.getRank() == 0)
			derVar = rewriter.create<LoadOp>(op->getLoc(), derVar);

		rewriter.replaceOp(op, derVar);

		return mlir::success();
	}

	private:
	Model* model;
};

struct SimulationOpPattern : public mlir::OpRewritePattern<SimulationOp>
{
	SimulationOpPattern(mlir::MLIRContext* context, SolveModelOptions options)
			: mlir::OpRewritePattern<SimulationOp>(context),
				options(options)
	{
	}

	mlir::LogicalResult matchAndRewrite(SimulationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location location = op->getLoc();

		llvm::SmallVector<mlir::Type, 3> varTypes;

		{
			auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());

			varTypes.push_back(terminator.args()[0].getType().cast<PointerType>().toUnknownAllocationScope());

			// Add the time step as second argument
			varTypes.push_back(op.timeStep().getType());

			for (auto it = ++terminator.args().begin(); it != terminator.args().end(); ++it)
				varTypes.push_back((*it).getType().cast<PointerType>().toUnknownAllocationScope());
		}

		auto structType = StructType::get(op->getContext(), varTypes);

		{
			// Init function
			auto functionType = rewriter.getFunctionType(llvm::None, structType);
			auto function = rewriter.create<mlir::FuncOp>(location, "init", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			rewriter.mergeBlocks(&op.init().front(), &function.body().front());

			llvm::SmallVector<mlir::Value, 3> values;
			auto terminator = mlir::cast<YieldOp>(entryBlock->getTerminator());

			auto removeAllocationScopeFn = [&](mlir::Value value) -> mlir::Value {
				return rewriter.create<PtrCastOp>(
						value.getLoc(), value,
						value.getType().cast<PointerType>().toUnknownAllocationScope());
			};

			// Time variable
			mlir::Value time = terminator.args()[0];
			values.push_back(removeAllocationScopeFn(time));

			// Time step
			mlir::Value timeStep = rewriter.create<ConstantOp>(op.getLoc(), op.timeStep());
			values.push_back(timeStep);

			// Add the remaining variables to the struct. Time and time step
			// variables have already been managed, but only the time one was in the
			// yield operation, so we need to start from its second argument.

			for (auto it = ++terminator.args().begin(); it != terminator.args().end(); ++it)
				values.push_back(removeAllocationScopeFn(*it));

			// Set the start time
			mlir::Value startTime = rewriter.create<ConstantOp>(location, op.startTime());
			rewriter.create<StoreOp>(op->getLoc(), startTime, values[0]);

			mlir::Value structValue = rewriter.create<PackOp>(terminator->getLoc(), values);

			rewriter.eraseOp(terminator);
			rewriter.create<mlir::ReturnOp>(location, structValue);
		}

		{
			// Step function
			auto functionType = rewriter.getFunctionType(structType, BooleanType::get(op.getContext()));
			auto function = rewriter.create<mlir::FuncOp>(location, "step", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value structValue = function.getArgument(0);
			llvm::SmallVector<mlir::Value, 3> args;

			args.push_back(rewriter.create<ExtractOp>(structValue.getLoc(), varTypes[0], structValue, 0));

			for (size_t i = 2, e = structType.getElementTypes().size(); i < e; ++i)
				args.push_back(rewriter.create<ExtractOp>(structValue.getLoc(), varTypes[i], structValue, i));

			// Check if the current time is less than the end time
			mlir::Value currentTime = args[0];
			currentTime = rewriter.create<LoadOp>(currentTime.getLoc(), currentTime);
			mlir::Value endTime = rewriter.create<ConstantOp>(location, op.endTime());
			mlir::Value condition = rewriter.create<LtOp>(location, BooleanType::get(op.getContext()), currentTime, endTime);

			auto ifOp = rewriter.create<IfOp>(function->getLoc(), BooleanType::get(rewriter.getContext()), condition, true);

			{
				// If we didn't reach the end time update the variables and return
				// true to continue the simulation;
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

				mlir::Value trueValue = rewriter.create<ConstantOp>(function.getLoc(), BooleanAttribute::get(BooleanType::get(op.getContext()), true));
				auto terminator = rewriter.create<YieldOp>(ifOp.getLoc(), trueValue);

				rewriter.eraseOp(op.body().front().getTerminator());
				rewriter.mergeBlockBefore(&op.body().front(), terminator, args);
			}

			{
				// Otherwise, return false to stop the simulation
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());

				mlir::Value falseValue = rewriter.create<ConstantOp>(function.getLoc(), BooleanAttribute::get(BooleanType::get(op.getContext()), false));
				rewriter.create<YieldOp>(ifOp.getLoc(), falseValue);
			}

			rewriter.create<mlir::ReturnOp>(function->getLoc(), ifOp.getResult(0));
		}

		{
			// Print function
			auto functionType = rewriter.getFunctionType(structType, llvm::None);
			auto function = rewriter.create<mlir::FuncOp>(location, "print", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value structValue = function.getArgument(0);
			mlir::BlockAndValueMapping mapping;

			mapping.map(op.print().getArgument(0), rewriter.create<ExtractOp>(structValue.getLoc(), varTypes[0], structValue, 0));

			for (size_t i = 2, e = structType.getElementTypes().size(); i < e; ++i)
				mapping.map(op.print().getArgument(i - 1), rewriter.create<ExtractOp>(structValue.getLoc(), varTypes[i], structValue, i));

			auto terminator = mlir::cast<YieldOp>(op.print().front().getTerminator());
			llvm::SmallVector<mlir::Value, 3> valuesToBePrinted;

			for (mlir::Value value : terminator.args())
				valuesToBePrinted.push_back(mapping.lookup(value));

			rewriter.create<PrintOp>(function.getLoc(), valuesToBePrinted);

			rewriter.create<mlir::ReturnOp>(function->getLoc());
		}

		if (options.emitMain)
		{
			auto functionType = rewriter.getFunctionType(llvm::None, llvm::None);
			auto function = rewriter.create<mlir::FuncOp>(location, "main", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value data = rewriter.create<CallOp>(function.getLoc(), "init", structType, llvm::None).getResult(0);

			// Create the simulation loop
			auto loop = rewriter.create<ForOp>(function.getLoc());

			{
				mlir::OpBuilder::InsertionGuard g(rewriter);

				rewriter.setInsertionPointToStart(&loop.condition().front());
				mlir::Value stepResult = rewriter.create<CallOp>(function->getLoc(), "step", BooleanType::get(rewriter.getContext()), data).getResult(0);
				rewriter.create<ConditionOp>(loop->getLoc(), stepResult);

				// The body contains just the print call, because the update is
				// already done by the step function in the condition region.
				rewriter.setInsertionPointToStart(&loop.body().front());
				rewriter.create<CallOp>(function->getLoc(), "print", llvm::None, data);
				rewriter.create<YieldOp>(loop->getLoc());

				// Increment the time
				rewriter.setInsertionPointToStart(&loop.step().front());
				mlir::Value structValue = data;

				mlir::Value time = rewriter.create<ExtractOp>(
						location,
						structValue.getType().cast<StructType>().getElementTypes()[0],
						structValue, 0);

				mlir::Value timeStep = rewriter.create<ExtractOp>(location, op.timeStep().getType(), structValue, 1);
				mlir::Value currentTime = rewriter.create<LoadOp>(location, time);

				mlir::Value increasedTime = rewriter.create<AddOp>(location, currentTime.getType(), currentTime, timeStep);
				rewriter.create<StoreOp>(location, increasedTime, time);

				rewriter.create<YieldOp>(location);
			}

			rewriter.create<mlir::ReturnOp>(function.getLoc());
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	SolveModelOptions options;
};

struct EquationOpPattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		// Create the assignment
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		rewriter.setInsertionPoint(sides);

		for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
		{
			if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
			{
				assert(loadOp.indexes().empty());
				rewriter.create<AssignmentOp>(sides.getLoc(), rhs, loadOp.memory());
			}
			else
			{
				rewriter.create<AssignmentOp>(sides->getLoc(), rhs, lhs);
			}
		}

		rewriter.eraseOp(sides);

		// Inline the equation body
		rewriter.setInsertionPoint(op);
		rewriter.mergeBlockBefore(op.body(), op);

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

struct ForEquationOpPattern : public mlir::OpRewritePattern<ForEquationOp>
{
	using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(ForEquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		// Create the assignment
		auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
		rewriter.setInsertionPoint(sides);

		for (auto [lhs, rhs] : llvm::zip(sides.lhs(), sides.rhs()))
		{
			if (auto loadOp = mlir::dyn_cast<LoadOp>(lhs.getDefiningOp()))
			{
				assert(loadOp.indexes().empty());
				rewriter.create<AssignmentOp>(sides.getLoc(), rhs, loadOp.memory());
			}
			else
			{
				rewriter.create<AssignmentOp>(sides->getLoc(), rhs, lhs);
			}
		}

		rewriter.eraseOp(sides);

		// Create the loop
		rewriter.setInsertionPoint(op);
		llvm::SmallVector<mlir::Block*, 3> bodies;
		llvm::SmallVector<mlir::Value, 3> inductions;
		mlir::Block* innermostBody = rewriter.getInsertionBlock();

		for (auto induction : op.inductionsDefinitions())
		{
			auto inductionOp = mlir::cast<InductionOp>(induction.getDefiningOp());

			// Init value
			mlir::Value start = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.start()));

			auto loop = rewriter.create<ForOp>(induction.getLoc(), start);
			bodies.push_back(&loop.body().front());
			inductions.push_back(loop.body().getArgument(0));

			{
				// Condition
				rewriter.setInsertionPointToStart(&loop.condition().front());
				mlir::Value end = rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(inductionOp.end()));
				mlir::Value current = loop.condition().getArgument(0);
				mlir::Value condition = rewriter.create<LteOp>(induction.getLoc(), BooleanType::get(op->getContext()), current, end);
				rewriter.create<ConditionOp>(induction.getLoc(), condition, loop.condition().getArgument(0));
			}

			{
				// Step
				rewriter.setInsertionPointToStart(&loop.step().front());
				mlir::Value newInductionValue = rewriter.create<AddOp>(induction.getLoc(),
																															 loop.step().getArgument(0).getType(),
																															 loop.step().getArgument(0),
																															 rewriter.create<ConstantOp>(induction.getLoc(), rewriter.getIndexAttr(1)));
				rewriter.create<YieldOp>(induction.getLoc(), newInductionValue);
			}

			// The next loop will be built inside the current body
			innermostBody = &loop.body().front();
			rewriter.setInsertionPointToStart(innermostBody);
		}

		rewriter.mergeBlocks(op.body(), innermostBody, inductions);

		// Add the terminator to each body block
		for (auto [block, induction] : llvm::zip(bodies, inductions))
		{
			rewriter.setInsertionPointToEnd(block);
			rewriter.create<YieldOp>(induction.getLoc(), induction);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}
};

/**
 * Model solver pass.
 * Its objective is to convert a descriptive (and thus not sequential) model
 * into an algorithmic one.
 */
class SolveModelPass: public mlir::PassWrapper<SolveModelPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit SolveModelPass(SolveModelOptions options)
			: options(std::move(options))
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		// Convert the scalar values into arrays of one element
		if (failed(loopify()))
			return signalPassFailure();

		// Scalarize the equations consisting in array assignments, by adding
		// the required inductions.
		if (failed(scalarizeArrayEquations()))
			return signalPassFailure();

		getOperation()->walk([&](SimulationOp simulation) {
			mlir::OpBuilder builder(simulation);

			// Create the model
			Model model = Model::build(simulation);

			// Remove the derivative operations and allocate the appropriate buffers
			if (failed(removeDerivatives(builder, model)))
				return signalPassFailure();

			// Match
			if (failed(match(model, options.matchingMaxIterations)))
				return signalPassFailure();

			// Solve circular dependencies
			if (failed(solveSCCs(builder, model, options.sccMaxIterations)))
				return signalPassFailure();

			// Schedule
			if (failed(schedule(model)))
				return signalPassFailure();

			// Explicitate the equations so that the updated variable is the only
			// one on the left-hand side of the equation.
			if (failed(explicitateEquations(model)))
				return signalPassFailure();

			// Calculate the values that the state variables will have in the next
			// iteration.
			if (failed(updateStates(builder, model)))
				return signalPassFailure();
		});

		// The model has been solved and we can now proceed to create the update
		// functions and, if requested, the main simulation loop.
		if (auto status = createSimulationFunctions(); failed(status))
			return signalPassFailure();
	}

	/**
	 * Convert the scalar variables to arrays with just one element and the
	 * scalar equations to for equations with a one-sized induction. This
	 * allows the subsequent transformation to ignore whether it was a scalar
	 * or a loop equation, and a later optimization pass can easily remove
	 * the new useless induction.
	 */
	mlir::LogicalResult loopify()
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<SimulationOp>([](SimulationOp op) {
			auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
			mlir::ValueRange args = terminator.args();

			for (auto it = std::next(args.begin()), end = args.end(); it != end; ++it)
				if (auto pointerType = (*it).getType().dyn_cast<PointerType>())
					if (pointerType.getRank() == 0)
						return false;

			return true;
		});

		mlir::OwningRewritePatternList patterns;
		patterns.insert<LoopifyPattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * If an equation consists in an assignment between two arrays, then
	 * convert it into a for equation, in order to scalarize the assignments.
	 */
	mlir::LogicalResult scalarizeArrayEquations()
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<EquationOp>([](EquationOp op) {
			auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
			auto pairs = llvm::zip(sides.lhs(), sides.rhs());

			return llvm::all_of(pairs, [](const auto& pair) {
				mlir::Type lhs = std::get<0>(pair).getType();
				mlir::Type rhs = std::get<1>(pair).getType();

				return !lhs.isa<PointerType>() && !rhs.isa<PointerType>();
			});
		});

		target.addDynamicallyLegalOp<ForEquationOp>([](ForEquationOp op) {
			auto sides = mlir::cast<EquationSidesOp>(op.body()->getTerminator());
			auto pairs = llvm::zip(sides.lhs(), sides.rhs());

			return llvm::all_of(pairs, [](const auto& pair) {
				mlir::Type lhs = std::get<0>(pair).getType();
				mlir::Type rhs = std::get<1>(pair).getType();

				return !lhs.isa<PointerType>() && !rhs.isa<PointerType>();
			});
		});

		mlir::OwningRewritePatternList patterns;

		patterns.insert<
		    EquationOpScalarizePattern,
				ForEquationOpScalarizePattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	/**
	 * Remove the derivative operations by replacing them with appropriate
	 * buffers, and set the derived variables as state variables.
	 *
	 * @param builder operation builder
	 * @param model   model
	 * @return conversion result
	 */
	mlir::LogicalResult removeDerivatives(mlir::OpBuilder& builder, Model& model)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<DerOp>();

		mlir::OwningRewritePatternList patterns;
		patterns.insert<DerOpPattern>(&getContext(), model);

		if (auto status = applyPartialConversion(model.getOp(), target, std::move(patterns)); failed(status))
			return status;

		model.reloadIR();
		return mlir::success();
	}

	mlir::LogicalResult explicitateEquations(Model& model)
	{
		for (auto& equation : model.getEquations())
			if (auto res = equation.explicitate(); failed(res))
				return res;

		return mlir::success();
	}

	mlir::LogicalResult updateStates(mlir::OpBuilder& builder, Model& model)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		mlir::Location location = model.getOp()->getLoc();

		builder.setInsertionPointToStart(&model.getOp().body().front());
		//builder.setInsertionPoint(model.getOp().body().back().getTerminator());
		mlir::Value timeStep = builder.create<ConstantOp>(location, model.getOp().timeStep());

		for (auto& variable : model.getVariables())
		{
			if (!variable->isState())
				continue;

			mlir::Value var = variable->getReference();
			mlir::Value der = variable->getDer();

			auto terminator = mlir::cast<YieldOp>(model.getOp().init().front().getTerminator());

			for (auto value : llvm::enumerate(terminator.args()))
			{
				if (value.value() == var)
					var = model.getOp().body().front().getArgument(value.index());

				if (value.value() == der)
					der = model.getOp().body().front().getArgument(value.index());
			}

			mlir::Value varReference = var;

			if (auto pointerType = var.getType().cast<PointerType>(); pointerType.getRank() == 0)
				var = builder.create<LoadOp>(location, var);

			if (auto pointerType = der.getType().cast<PointerType>(); pointerType.getRank() == 0)
				der = builder.create<LoadOp>(location, der);

			mlir::Value newValue = builder.create<MulOp>(location, der.getType(), der, timeStep);
			newValue = builder.create<AddOp>(location, var.getType(), newValue, var);
			builder.create<AssignmentOp>(location, newValue, varReference);
		}

		return mlir::success();
	}

	mlir::LogicalResult createSimulationFunctions()
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<SimulationOp, EquationOp, ForEquationOp>();

		mlir::OwningRewritePatternList patterns;
		patterns.insert<SimulationOpPattern>(&getContext(), options);
		patterns.insert<EquationOpPattern, ForEquationOpPattern>(&getContext());

		if (auto status = applyPartialConversion(getOperation(), target, std::move(patterns)); failed(status))
			return status;

		return mlir::success();
	}

	private:
	SolveModelOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createSolveModelPass(SolveModelOptions options)
{
	return std::make_unique<SolveModelPass>(options);
}
