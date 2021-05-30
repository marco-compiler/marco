#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/SolveModel.h>
#include <modelica/mlirlowerer/passes/matching/Matching.h>
#include <modelica/mlirlowerer/passes/matching/SCCCollapsing.h>
#include <modelica/mlirlowerer/passes/matching/Schedule.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>

using namespace modelica;
using namespace codegen;
using namespace model;

struct SimulationOpLoopifyPattern : public mlir::OpRewritePattern<SimulationOp>
{
	using mlir::OpRewritePattern<SimulationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SimulationOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
		rewriter.eraseOp(terminator);

		llvm::SmallVector<mlir::Value, 3> newVars;
		newVars.push_back(terminator.values()[0]);

		unsigned int index = 1;

		std::map<ForEquationOp, mlir::Value> inductions;

		for (auto it = std::next(terminator.values().begin()); it != terminator.values().end(); ++it)
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

					for (auto useIt = originalArgument.use_begin(), end = originalArgument.use_end(); useIt != end;)
					{
						auto& use = *useIt;
						++useIt;
						rewriter.setInsertionPoint(use.getOwner());

						auto* parentEquation = use.getOwner()->getParentWithTrait<EquationInterface::Trait>();

						if (auto equationOp = mlir::dyn_cast<EquationOp>(parentEquation))
						{
							auto forEquationOp = convertToForEquation(rewriter, equationOp.getLoc(), equationOp);
							inductions[forEquationOp] = forEquationOp.body()->getArgument(0);
						}
					}

					for (auto useIt = originalArgument.use_begin(), end = originalArgument.use_end(); useIt != end;)
					{
						auto& use = *useIt;
						++useIt;
						rewriter.setInsertionPoint(use.getOwner());

						auto* parentEquation = use.getOwner()->getParentWithTrait<EquationInterface::Trait>();
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
		auto result = rewriter.create<SimulationOp>(op->getLoc(), op.startTime(), op.endTime(), op.timeStep(), op.body().getArgumentTypes());
		rewriter.mergeBlocks(&op.init().front(), &result.init().front(), result.init().getArguments());
		rewriter.mergeBlocks(&op.body().front(), &result.body().front(), result.body().getArguments());
		rewriter.mergeBlocks(&op.print().front(), &result.print().front(), result.print().getArguments());

		//rewriter.setInsertionPointToStart(&result.body().front());
		//rewriter.create<YieldOp>(op->getLoc());

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	static ForEquationOp convertToForEquation(mlir::PatternRewriter& rewriter, mlir::Location loc, EquationOp equation)
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

struct EquationOpLoopifyPattern : public mlir::OpRewritePattern<EquationOp>
{
	using mlir::OpRewritePattern<EquationOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(EquationOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();

		auto forEquation = rewriter.create<ForEquationOp>(loc, 1);

		// Inductions
		rewriter.setInsertionPointToStart(forEquation.inductionsBlock());
		mlir::Value induction = rewriter.create<InductionOp>(loc, 1, 1);
		rewriter.create<YieldOp>(loc, induction);

		// Body
		rewriter.mergeBlocks(op.body(), forEquation.body());

		forEquation.body()->walk([&](SubscriptionOp subscription) {
			if (subscription.indexes().size() == 1)
			{
				auto index = subscription.indexes()[0].getDefiningOp<ConstantOp>();
				auto indexValue = index.value().cast<IntegerAttribute>().getValue();
				rewriter.setInsertionPoint(subscription);

				mlir::Value newIndex = rewriter.create<AddOp>(
						subscription->getLoc(), rewriter.getIndexType(), forEquation.induction(0),
						rewriter.create<ConstantOp>(subscription->getLoc(), rewriter.getIndexAttr(indexValue - 1)));

				rewriter.replaceOp(index, newIndex);
			}
		});

		rewriter.eraseOp(op);
		return mlir::success();
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
	DerOpPattern(mlir::MLIRContext* context, Model& model, mlir::BlockAndValueMapping& derivatives)
			: mlir::OpRewritePattern<DerOp>(context), model(&model), derivatives(&derivatives)
	{
	}

	mlir::LogicalResult matchAndRewrite(DerOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Location loc = op->getLoc();
		mlir::Value operand = op.operand();

		// If the value to be derived belongs to an array, then also the derived
		// value is stored within an array. Thus we need to store its position.

		llvm::SmallVector<mlir::Value, 3> subscriptions;

		while (!operand.isa<mlir::BlockArgument>())
		{
			if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(operand.getDefiningOp()))
			{
				mlir::ValueRange indexes = subscriptionOp.indexes();
				subscriptions.append(indexes.begin(), indexes.end());
				operand = subscriptionOp.source();
			}
			else
			{
				return rewriter.notifyMatchFailure(op, "Unexpected operation");
			}
		}

		auto simulation = op->getParentOfType<SimulationOp>();
		mlir::Value var = simulation.getVariableAllocation(operand);
		auto variable = model->getVariable(var);
		mlir::Value derVar;

		if (!derivatives->contains(operand))
		{
			auto terminator = mlir::cast<YieldOp>(var.getParentBlock()->getTerminator());
			rewriter.setInsertionPointAfter(terminator);

			llvm::SmallVector<mlir::Value, 3> args;

			for (mlir::Value arg : terminator.values())
				args.push_back(arg);

			if (auto pointerType = variable.getReference().getType().dyn_cast<PointerType>())
				derVar = rewriter.create<AllocOp>(loc, pointerType.getElementType(), pointerType.getShape(), llvm::None, false);
			else
			{
				derVar = rewriter.create<AllocOp>(loc, variable.getReference().getType(), llvm::None, llvm::None, false);
			}

			model->addVariable(derVar);
			variable.setDer(derVar);

			args.push_back(derVar);
			rewriter.create<YieldOp>(terminator.getLoc(), args);
			rewriter.eraseOp(terminator);

			auto newArgumentType = derVar.getType().cast<PointerType>().toUnknownAllocationScope();
			auto bodyArgument = simulation.body().addArgument(newArgumentType);
			simulation.print().addArgument(newArgumentType);

			derivatives->map(operand, bodyArgument);
		}
		else
		{
			derVar = variable.getDer();
		}

		rewriter.setInsertionPoint(op);

		op.dump();
		operand.dump();
		derVar = derivatives->lookup(operand);

		if (!subscriptions.empty())
			derVar = rewriter.create<SubscriptionOp>(loc, derVar, subscriptions);

		if (auto pointerType = derVar.getType().cast<PointerType>(); pointerType.getRank() == 0)
			derVar = rewriter.create<LoadOp>(loc, derVar);

		rewriter.replaceOp(op, derVar);

		return mlir::success();
	}

	private:
	Model* model;
	mlir::BlockAndValueMapping* derivatives;
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
		mlir::Location loc = op->getLoc();

		llvm::SmallVector<mlir::Type, 3> varTypes;

		{
			auto terminator = mlir::cast<YieldOp>(op.init().back().getTerminator());
			varTypes.push_back(terminator.values()[0].getType().cast<PointerType>().toUnknownAllocationScope());

			// Add the time step as second argument
			varTypes.push_back(op.timeStep().getType());

			for (auto it = ++terminator.values().begin(); it != terminator.values().end(); ++it)
				varTypes.push_back((*it).getType().cast<PointerType>().toUnknownAllocationScope());
		}

		auto structType = StructType::get(op->getContext(), varTypes);
		auto structPtrType = PointerType::get(structType.getContext(), BufferAllocationScope::unknown, structType);
		auto opaquePtrType = OpaquePointerType::get(structPtrType.getContext());

		{
			// Init function
			auto functionType = rewriter.getFunctionType(llvm::None, opaquePtrType);
			auto function = rewriter.create<mlir::FuncOp>(loc, "init", functionType);
			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			rewriter.mergeBlocks(&op.init().front(), &function.body().front());

			llvm::SmallVector<mlir::Value, 3> values;
			auto terminator = mlir::cast<YieldOp>(entryBlock->getTerminator());

			auto removeAllocationScopeFn = [&](mlir::Value value) -> mlir::Value {
				return rewriter.create<PtrCastOp>(
						loc, value,
						value.getType().cast<PointerType>().toUnknownAllocationScope());
			};

			// Time variable
			mlir::Value time = terminator.values()[0];
			values.push_back(removeAllocationScopeFn(time));

			// Time step
			mlir::Value timeStep = rewriter.create<ConstantOp>(loc, op.timeStep());
			values.push_back(timeStep);

			// Add the remaining variables to the struct. Time and time step
			// variables have already been managed, but only the time one was in the
			// yield operation, so we need to start from its second argument.

			for (auto it = std::next(terminator.values().begin()); it != terminator.values().end(); ++it)
				values.push_back(removeAllocationScopeFn(*it));

			// Set the start time
			mlir::Value startTime = rewriter.create<ConstantOp>(loc, op.startTime());
			rewriter.create<StoreOp>(loc, startTime, values[0]);

			mlir::Value structValue = rewriter.create<PackOp>(terminator->getLoc(), values);
			mlir::Value result = rewriter.create<AllocOp>(structValue.getLoc(), structType, llvm::None, llvm::None, false);
			rewriter.create<StoreOp>(result.getLoc(), structValue, result);
			result = rewriter.create<PtrCastOp>(result.getLoc(), result, opaquePtrType);

			rewriter.replaceOpWithNewOp<mlir::ReturnOp>(terminator, result);
		}

		{
			// Step function
			auto function = rewriter.create<mlir::FuncOp>(
					loc, "step",
					rewriter.getFunctionType(opaquePtrType, BooleanType::get(op->getContext())));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value structValue = loadDataFromOpaquePtr(rewriter, loc, function.getArgument(0), structType);

			llvm::SmallVector<mlir::Value, 3> args;
			args.push_back(rewriter.create<ExtractOp>(loc, varTypes[0], structValue, 0));

			for (size_t i = 2, e = structType.getElementTypes().size(); i < e; ++i)
				args.push_back(rewriter.create<ExtractOp>(loc, varTypes[i], structValue, i));

			// Check if the current time is less than the end time
			mlir::Value currentTime = rewriter.create<LoadOp>(loc, args[0]);
			mlir::Value endTime = rewriter.create<ConstantOp>(loc, op.endTime());

			mlir::Value condition = rewriter.create<LtOp>(
					loc, BooleanType::get(op->getContext()), currentTime, endTime);

			auto ifOp = rewriter.create<IfOp>(loc, BooleanType::get(op->getContext()), condition, true);

			{
				// If we didn't reach the end time update the variables and return
				// true to continue the simulation.
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

				mlir::Value trueValue = rewriter.create<ConstantOp>(loc, getBooleanAttribute(op->getContext(), true));
				auto terminator = rewriter.create<YieldOp>(loc, trueValue);

				rewriter.eraseOp(op.body().front().getTerminator());
				rewriter.mergeBlockBefore(&op.body().front(), terminator, args);
			}

			{
				// Otherwise, return false to stop the simulation
				mlir::OpBuilder::InsertionGuard g(rewriter);
				rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());

				mlir::Value falseValue = rewriter.create<ConstantOp>(loc, getBooleanAttribute(op->getContext(), false));
				rewriter.create<YieldOp>(loc, falseValue);
			}

			rewriter.create<mlir::ReturnOp>(loc, ifOp.getResult(0));
		}

		{
			// Print function
			auto function = rewriter.create<mlir::FuncOp>(
					loc, "print",
					rewriter.getFunctionType(opaquePtrType, llvm::None));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			mlir::Value structValue = loadDataFromOpaquePtr(rewriter, loc, function.getArgument(0), structType);
			mlir::BlockAndValueMapping mapping;

			mapping.map(op.print().getArgument(0), rewriter.create<ExtractOp>(loc, varTypes[0], structValue, 0));

			for (size_t i = 2, e = structType.getElementTypes().size(); i < e; ++i)
				mapping.map(op.print().getArgument(i - 1), rewriter.create<ExtractOp>(loc, varTypes[i], structValue, i));

			auto terminator = mlir::cast<YieldOp>(op.print().front().getTerminator());
			llvm::SmallVector<mlir::Value, 3> valuesToBePrinted;

			for (mlir::Value value : terminator.values())
				valuesToBePrinted.push_back(mapping.lookup(value));

			rewriter.create<PrintOp>(loc, valuesToBePrinted);

			rewriter.create<mlir::ReturnOp>(loc);
		}

		if (options.emitMain)
		{
			// The main function takes care of running the simulation loop. More
			// precisely, it first calls the "init" function, and then keeps
			// running the updates until the step function return the stop
			// condition (that is, a false value). After each step, it also
			// prints the values and increments the time.

			auto function = rewriter.create<mlir::FuncOp>(
					loc, "main", rewriter.getFunctionType(llvm::None, llvm::None));

			auto* entryBlock = function.addEntryBlock();

			mlir::OpBuilder::InsertionGuard guard(rewriter);
			rewriter.setInsertionPointToStart(entryBlock);

			// Initialize the variables
			mlir::Value data = rewriter.create<CallOp>(loc, "init", opaquePtrType, llvm::None).getResult(0);

			// Create the simulation loop
			auto loop = rewriter.create<ForOp>(loc);

			{
				mlir::OpBuilder::InsertionGuard g(rewriter);

				rewriter.setInsertionPointToStart(&loop.condition().front());
				mlir::Value shouldContinue = rewriter.create<CallOp>(loc, "step", BooleanType::get(op->getContext()), data).getResult(0);
				rewriter.create<ConditionOp>(loc, shouldContinue);

				// The body contains just the print call, because the update is
				// already done by the "step "function in the condition region.
				// Note that the update is not done if the "step" function detects
				// that the simulation has already come to the end.

				rewriter.setInsertionPointToStart(&loop.body().front());
				rewriter.create<CallOp>(loc, "print", llvm::None, data);
				rewriter.create<YieldOp>(loc);

				// Increment the time
				rewriter.setInsertionPointToStart(&loop.step().front());
				mlir::Value structValue = loadDataFromOpaquePtr(rewriter, loc, data, structType);

				mlir::Value time = rewriter.create<ExtractOp>(
						loc,
						structValue.getType().cast<StructType>().getElementTypes()[0],
						structValue, 0);

				mlir::Value timeStep = rewriter.create<ExtractOp>(
						loc, op.timeStep().getType(), structValue, 1);

				mlir::Value currentTime = rewriter.create<LoadOp>(loc, time);

				mlir::Value increasedTime = rewriter.create<AddOp>(
						loc, currentTime.getType(), currentTime, timeStep);

				rewriter.create<StoreOp>(loc, increasedTime, time);
				rewriter.create<YieldOp>(loc);
			}

			rewriter.create<mlir::ReturnOp>(loc);
		}

		rewriter.eraseOp(op);
		return mlir::success();
	}

	private:
	static BooleanAttribute getBooleanAttribute(mlir::MLIRContext* context, bool value)
	{
		mlir::Type booleanType = BooleanType::get(context);
		return BooleanAttribute::get(booleanType, value);
	}

	static mlir::Value loadDataFromOpaquePtr(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value ptr, StructType structType)
	{
		assert(ptr.getType().isa<OpaquePointerType>());
		mlir::Type structPtrType = PointerType::get(structType.getContext(), BufferAllocationScope::unknown, structType);
		mlir::Value castedPtr = builder.create<PtrCastOp>(loc, ptr, structPtrType);
		return builder.create<LoadOp>(loc, castedPtr);
	}

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
		// TODO: addTimeDerivative

		// Convert the scalar values into arrays of one element
		if (failed(loopify()))
			return signalPassFailure();

		// Scalarize the equations consisting in array assignments, by adding
		// the required inductions.
		if (failed(scalarizeArrayEquations()))
			return signalPassFailure();

		getOperation()->dump();

		getOperation()->walk([&](SimulationOp simulation) {
			mlir::OpBuilder builder(simulation);

			// Create the model
			Model model = Model::build(simulation);
			mlir::BlockAndValueMapping derivatives;

			// Remove the derivative operations and allocate the appropriate buffers
			if (failed(removeDerivatives(builder, model, derivatives)))
				return signalPassFailure();

			if (failed(instantiateFunctionDerivatives(builder, model, derivatives)))
				return signalPassFailure();

			// Match
			if (failed(match(model, options.matchingMaxIterations)))
				return signalPassFailure();

			// Solve circular dependencies
			if (failed(solveSCCs(builder, model, options.sccMaxIterations)))
				return signalPassFailure();

			simulation->dump();

			// Schedule
			if (failed(schedule(model)))
				return signalPassFailure();

			simulation->dump();

			// Explicitate the equations so that the updated variable is the only
			// one on the left-hand side of the equation.
			if (failed(explicitateEquations(model)))
				return signalPassFailure();

			simulation->dump();

			// Select and use the solver
			if (failed(selectSolver(builder, model)))
				return signalPassFailure();

			simulation.dump();
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
			mlir::ValueRange args = terminator.values();

			for (auto it = std::next(args.begin()), end = args.end(); it != end; ++it)
				if (auto pointerType = (*it).getType().dyn_cast<PointerType>())
					if (pointerType.getRank() == 0)
						return false;

			return true;
		});

		target.addIllegalOp<EquationOp>();

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<SimulationOpLoopifyPattern, EquationOpLoopifyPattern>(&getContext());
		patterns.insert<SimulationOpLoopifyPattern>(&getContext());

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

		mlir::OwningRewritePatternList patterns(&getContext());

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
	mlir::LogicalResult removeDerivatives(mlir::OpBuilder& builder, Model& model, mlir::BlockAndValueMapping& derivatives)
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<DerOp>();

		mlir::OwningRewritePatternList patterns(&getContext());
		patterns.insert<DerOpPattern>(&getContext(), model, derivatives);

		if (auto status = applyPartialConversion(model.getOp(), target, std::move(patterns)); failed(status))
			return status;

		model.reloadIR();
		return mlir::success();
	}

	mlir::LogicalResult instantiateFunctionDerivatives(mlir::OpBuilder& builder, Model& model, mlir::BlockAndValueMapping& derivatives)
	{
		/*
		model.getOp()->walk([&](ForEquationOp equation) {
			equation.walk([&](CallOp call) {
				auto module = call->getParentOfType<mlir::ModuleOp>();
				auto simulation = call->getParentOfType<SimulationOp>();
				auto callee = module.lookupSymbol<FunctionOp>(call.callee());

				if (!callee.hasDerivative())
					return;

				builder.setInsertionPointAfter(equation);
				auto derivedEquation = builder.create<ForEquationOp>(equation->getLoc(), equation.inductions().size());

				llvm::SmallVector<mlir::Value, 3> lhs;
				llvm::SmallVector<mlir::Value, 3> rhs;

				for (mlir::Value lhsValue : equation.lhs())
				{
					if (mlir::isa<mlir::BlockArgument>(lhsValue))
					{
						auto var = model.getVariable(simulation.getVariableAllocation(lhsValue));

						if (!var.isState())
							return;

						lhs.push_back()
					}
				}
			});
		});

		model.reloadIR();
		 */
		return mlir::success();
	}

	mlir::LogicalResult explicitateEquations(Model& model)
	{
		for (auto& equation : model.getEquations())
			if (auto status = equation.explicitate(); failed(status))
				return status;

		return mlir::success();
	}

	mlir::LogicalResult selectSolver(mlir::OpBuilder& builder, Model& model)
	{
		if (options.solverName == ForwardEuler)
			return updateStates(builder, model);

		if (options.solverName == CleverDAE)
			return addBltBlocks(builder, model);

		return mlir::failure();
	}

	/**
	 * Calculate the values that the state variables will have in the next
	 * iteration.
	 */
	mlir::LogicalResult updateStates(mlir::OpBuilder& builder, Model& model)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		mlir::Location loc = model.getOp()->getLoc();

		// Theoretically, given a state variable x and the current step time n,
		// the value of x(n + 1) should be determined at the end of the step of
		// time n, and the assignment of the new value should be done at time
		// n + 1. Anyway, doing so would require to create an additional buffer
		// to store x(n + 1), so that it can be assigned at the beginning of step
		// n + 1. This allocation can be avoided by computing x(n + 1) right at
		// the beginning of step n + 1, when the derivatives still have the values
		// of step n.

		builder.setInsertionPointToStart(&model.getOp().body().front());

		mlir::Value timeStep = builder.create<ConstantOp>(loc, model.getOp().timeStep());

		for (auto& variable : model.getVariables())
		{
			if (!variable->isState())
				continue;

			if (variable->isTime())
				continue;

			mlir::Value var = variable->getReference();
			mlir::Value der = variable->getDer();

			auto terminator = mlir::cast<YieldOp>(model.getOp().init().front().getTerminator());

			for (auto value : llvm::enumerate(terminator.values()))
			{
				if (value.value() == var)
					var = model.getOp().body().front().getArgument(value.index());

				if (value.value() == der)
					der = model.getOp().body().front().getArgument(value.index());
			}

			mlir::Value nextValue = builder.create<MulOp>(loc, der.getType(), der, timeStep);
			nextValue = builder.create<AddOp>(loc, var.getType(), nextValue, var);
			builder.create<AssignmentOp>(loc, nextValue, var);
		}

		return mlir::success();
	}

	/**
	 * This method transforms all differential equations and implicit equations
	 * into BLT blocks within the model. Then the assigned model, ready for
	 * lowering, is returned.
	 */
	mlir::LogicalResult addBltBlocks(mlir::OpBuilder& builder, Model& model)
	{
		assert(false && "To be implemented");
	}

	mlir::LogicalResult createSimulationFunctions()
	{
		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });
		target.addIllegalOp<SimulationOp, EquationOp, ForEquationOp>();

		mlir::OwningRewritePatternList patterns(&getContext());
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
