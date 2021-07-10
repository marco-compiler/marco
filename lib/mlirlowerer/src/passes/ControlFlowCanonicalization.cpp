#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ControlFlowCanonicalization.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>

using namespace modelica::codegen;

class BreakRemover
{
	public:
	BreakRemover(mlir::TypeConverter& typeConverter) : typeConverter(&typeConverter)
	{
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, mlir::Operation* function)
	{
		assert(function->hasTrait<mlir::OpTrait::FunctionLike>());
		auto& body = mlir::impl::getFunctionBody(function);
		llvm::SmallVector<mlir::Operation*, 3> bodyOps;

		for (auto& bodyOp : body.getOps())
			bodyOps.push_back(&bodyOp);

		for (auto& bodyOp : bodyOps)
			run(builder, bodyOp, []() -> mlir::Value {
				assert(false && "Operation has no parent stop condition");
				return nullptr;
			});

		return mlir::success();
	}

	private:
	bool run(mlir::OpBuilder& builder, mlir::Operation* op, std::function<mlir::Value()> stopCondition)
	{
		if (auto breakOp = mlir::dyn_cast<BreakOp>(op))
			return run(builder, breakOp, stopCondition);

		if (auto forOp = mlir::dyn_cast<ForOp>(op))
			return run(builder, forOp);

		if (auto ifOp = mlir::dyn_cast<IfOp>(op))
			return run(builder, ifOp, stopCondition);

		if (auto whileOp = mlir::dyn_cast<WhileOp>(op))
			return run(builder, whileOp);

		return false;
	}

	bool run(mlir::OpBuilder& builder, BreakOp op, std::function<mlir::Value()> stopCondition)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		mlir::Value trueValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(true));
		builder.create<mlir::memref::StoreOp>(op->getLoc(), trueValue, stopCondition());

		op.erase();
		return true;
	}

	bool run(mlir::OpBuilder& builder, ForOp op)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);

		mlir::Value stopConditionMemRef = nullptr;

		auto stopCondition = [&]() -> mlir::Value {
			if (stopConditionMemRef != nullptr)
				return stopConditionMemRef;

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPoint(op);
			stopConditionMemRef = builder.create<mlir::memref::AllocaOp>(op->getLoc(), mlir::MemRefType::get(llvm::None, builder.getI1Type()));
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::memref::StoreOp>(op->getLoc(), falseValue, stopConditionMemRef);

			return stopConditionMemRef;
		};

		// We need to make a copy of the operations list, or we would otherwise
		// invalidate the iterator as we change the IR structure.
		llvm::SmallVector<mlir::Operation*, 3> originalStatements;

		for (auto& statement : op.body().getOps())
			originalStatements.push_back(&statement);

		bool breakable = false;
		llvm::SmallVector<mlir::Operation*, 3> statements;
		llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

		for (auto& statement : originalStatements)
		{
			bool currentBreakable = run(builder, statement, stopCondition);

			if (breakable)
				avoidableStatements.push_back(statement);
			else
				statements.push_back(statement);

			breakable |= currentBreakable;
		}

		if (breakable)
		{
			// The loop is supposed to be breakable. Thus, before checking the normal
			// condition, we first need to check if the break condition variable has
			// been set to true in the previous loop execution. If it is set to true,
			// it means that a break statement has been executed and thus the loop
			// must be terminated.

			// Keep track of the operations used to determine the old condition.
			llvm::SmallVector<mlir::Operation*, 3> oldConditionOps;

			for (auto& oldConditionOp : op.condition().getOps())
				oldConditionOps.push_back(&oldConditionOp);

			// Check the condition
			builder.setInsertionPointToStart(&op.condition().front());
			mlir::Value condition = builder.create<mlir::memref::LoadOp>(op->getLoc(), stopCondition());
			auto ifOp = builder.create<mlir::scf::IfOp>(op->getLoc(), builder.getI1Type(), condition, true);

			// If the break condition variable is set to true, return false from the
			// condition block in order to stop the loop execution.
			builder.setInsertionPointToStart(&ifOp.thenRegion().front());
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::scf::YieldOp>(op->getLoc(), falseValue);

			// Move the original condition check in the "else" branch
			for (auto& oldConditionOp : oldConditionOps)
				oldConditionOp->moveBefore(&ifOp.elseRegion().front(), ifOp.elseRegion().front().getOperations().end());

			auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
			builder.setInsertionPointAfter(conditionOp);
			mlir::Value oldCondition = typeConverter->materializeTargetConversion(builder, op->getLoc(), builder.getI1Type(), conditionOp.condition());
			builder.create<mlir::scf::YieldOp>(conditionOp->getLoc(), oldCondition);
			conditionOp->erase();

			// Return the new condition
			builder.setInsertionPointAfter(ifOp);
			mlir::Value newCondition = typeConverter->materializeSourceConversion(builder, op->getLoc(), BooleanType::get(op->getContext()), ifOp.getResult(0));
			builder.create<ConditionOp>(op->getLoc(), newCondition);
		}

		// Create the block of code to be executed if a break has not been called
		if (breakable && !avoidableStatements.empty())
			wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

		// A while statement can't break a parent one
		return false;
	}

	bool run(mlir::OpBuilder& builder, IfOp op, std::function<mlir::Value()> stopCondition)
	{
		bool breakable = false;

		llvm::SmallVector<mlir::Region*, 3> regions;
		regions.push_back(&op.thenRegion());
		regions.push_back(&op.elseRegion());

		for (auto* region : regions)
		{
			// We need to make a copy of the operations list, or we would otherwise
			// invalidate the iterator as we change the IR structure.
			llvm::SmallVector<mlir::Operation*, 3> originalStatements;

			for (auto& statement : region->getOps())
				originalStatements.push_back(&statement);

			bool regionBreakable = false;
			llvm::SmallVector<mlir::Operation*, 3> statements;
			llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

			for (auto& statement : originalStatements)
			{
				bool currentBreakable = run(builder, statement, stopCondition);

				if (regionBreakable)
					avoidableStatements.push_back(statement);
				else
					statements.push_back(statement);

				regionBreakable |= currentBreakable;
			}

			if (breakable && !avoidableStatements.empty())
				wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

			breakable |= regionBreakable;
		}

		return breakable;
	}

	bool run(mlir::OpBuilder& builder, WhileOp op)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);

		mlir::Value stopConditionMemRef = nullptr;

		auto stopCondition = [&]() -> mlir::Value {
			if (stopConditionMemRef != nullptr)
				return stopConditionMemRef;

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPoint(op);
			stopConditionMemRef = builder.create<mlir::memref::AllocaOp>(op->getLoc(), mlir::MemRefType::get(llvm::None, builder.getI1Type()));
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::memref::StoreOp>(op->getLoc(), falseValue, stopConditionMemRef);

			return stopConditionMemRef;
		};

		bool breakable = false;
		llvm::SmallVector<mlir::Operation*, 3> statements;
		llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

		// We need to make a copy of the operations list, or we would otherwise
		// invalidate the iterator as we change the IR structure.
		llvm::SmallVector<mlir::Operation*, 3> originalStatements;

		for (auto& statement : op.body().getOps())
			originalStatements.push_back(&statement);

		for (auto& statement : originalStatements)
		{
			bool currentBreakable = run(builder, statement, stopCondition);

			if (breakable)
				avoidableStatements.push_back(statement);
			else
				statements.push_back(statement);

			breakable |= currentBreakable;
		}

		if (breakable)
		{
			// The loop is supposed to be breakable. Thus, before checking the normal
			// condition, we first need to check if the break condition variable has
			// been set to true in the previous loop execution. If it is set to true,
			// it means that a break statement has been executed and thus the loop
			// must be terminated.

			// Keep track of the operations used to determine the old condition.
			llvm::SmallVector<mlir::Operation*, 3> oldConditionOps;

			for (auto& oldConditionOp : op.condition().getOps())
				oldConditionOps.push_back(&oldConditionOp);

			// Check the condition
			builder.setInsertionPointToStart(&op.condition().front());
			mlir::Value condition = builder.create<mlir::memref::LoadOp>(op->getLoc(), stopCondition());
			auto ifOp = builder.create<mlir::scf::IfOp>(op->getLoc(), builder.getI1Type(), condition, true);

			// If the break condition variable is set to true, return false from the
			// condition block in order to stop the loop execution.
			builder.setInsertionPointToStart(&ifOp.thenRegion().front());
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::scf::YieldOp>(op->getLoc(), falseValue);

			// Move the original condition check in the "else" branch
			for (auto& oldConditionOp : oldConditionOps)
				oldConditionOp->moveBefore(&ifOp.elseRegion().front(), ifOp.elseRegion().front().getOperations().end());

			auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
			builder.setInsertionPointAfter(conditionOp);
			mlir::Value oldCondition = typeConverter->materializeTargetConversion(builder, op->getLoc(), builder.getI1Type(), conditionOp.condition());
			builder.create<mlir::scf::YieldOp>(conditionOp->getLoc(), oldCondition);
			conditionOp->erase();

			// Return the new condition
			builder.setInsertionPointAfter(ifOp);
			mlir::Value newCondition = typeConverter->materializeSourceConversion(builder, op->getLoc(), BooleanType::get(op->getContext()), ifOp.getResult(0));
			builder.create<ConditionOp>(op->getLoc(), newCondition);
		}

		// Create the block of code to be executed if a break has not been called
		if (breakable && !avoidableStatements.empty())
			wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

		// A while statement can't break a parent one
		return false;
	}

	void wrapAvoidableOps(mlir::OpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Operation*> ops, std::function<mlir::Value()> stopCondition) const
	{
		if (ops.empty())
			return;

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(ops[0]);

		mlir::Value condition = builder.create<mlir::memref::LoadOp>(loc, stopCondition());
		mlir::Value falseValue = builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(false));
		condition = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, condition, falseValue);
		auto ifOp = builder.create<mlir::scf::IfOp>(loc, condition);

		mlir::BlockAndValueMapping mapping;
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (auto& op : ops)
		{
			builder.clone(*op, mapping);
			op->erase();
		}
	}

	mlir::TypeConverter* typeConverter;
};

class ReturnRemover
{
	public:
	ReturnRemover(mlir::TypeConverter& typeConverter) : typeConverter(&typeConverter)
	{
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, mlir::Operation* function)
	{
		assert(function->hasTrait<mlir::OpTrait::FunctionLike>());
		auto& body = mlir::impl::getFunctionBody(function);
		llvm::SmallVector<mlir::Operation*, 3> bodyOps;

		for (auto& bodyOp : body.getOps())
			bodyOps.push_back(&bodyOp);

		mlir::Value stopConditionMemRef = nullptr;

		auto stopCondition = [&]() -> mlir::Value {
			if (stopConditionMemRef != nullptr)
				return stopConditionMemRef;

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPointToStart(&body.front());
			stopConditionMemRef = builder.create<mlir::memref::AllocaOp>(function->getLoc(), mlir::MemRefType::get(llvm::None, builder.getI1Type()));
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(function->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::memref::StoreOp>(function->getLoc(), falseValue, stopConditionMemRef);

			return stopConditionMemRef;
		};

		bool canReturn = false;
		llvm::SmallVector<mlir::Operation*, 3> statements;
		llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

		for (auto& statement : bodyOps)
		{
			bool currentCanReturn = run(builder, statement, stopCondition);

			if (canReturn)
				avoidableStatements.push_back(statement);
			else
				statements.push_back(statement);

			canReturn |= currentCanReturn;
		}

		if (canReturn && !avoidableStatements.empty())
			wrapAvoidableOps(builder, function->getLoc(), avoidableStatements, stopCondition);

		return mlir::success();
	}

	private:
	bool run(mlir::OpBuilder& builder, mlir::Operation* op, std::function<mlir::Value()> stopCondition)
	{
		if (auto earlyReturnOp = mlir::dyn_cast<ReturnOp>(op))
			return run(builder, earlyReturnOp, stopCondition);

		if (auto forOp = mlir::dyn_cast<ForOp>(op))
			return run(builder, forOp, stopCondition);

		if (auto ifOp = mlir::dyn_cast<IfOp>(op))
			return run(builder, ifOp, stopCondition);

		if (auto whileOp = mlir::dyn_cast<WhileOp>(op))
			return run(builder, whileOp, stopCondition);

		return false;
	}

	bool run(mlir::OpBuilder& builder, ReturnOp op, std::function<mlir::Value()> stopCondition)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		mlir::Value trueValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(true));
		builder.create<mlir::memref::StoreOp>(op->getLoc(), trueValue, stopCondition());

		op.erase();
		return true;
	}

	bool run(mlir::OpBuilder& builder, ForOp op, std::function<mlir::Value()> stopCondition)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);

		// We need to make a copy of the operations list, or we would otherwise
		// invalidate the iterator as we change the IR structure.
		llvm::SmallVector<mlir::Operation*, 3> originalStatements;

		for (auto& statement : op.body().getOps())
			originalStatements.push_back(&statement);

		bool canReturn = false;
		llvm::SmallVector<mlir::Operation*, 3> statements;
		llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

		for (auto& statement : originalStatements)
		{
			bool currentCanReturn = run(builder, statement, stopCondition);

			if (canReturn)
				avoidableStatements.push_back(statement);
			else
				statements.push_back(statement);

			canReturn |= currentCanReturn;
		}

		if (canReturn)
		{
			// The loop is supposed to be breakable. Thus, before checking the normal
			// condition, we first need to check if the break condition variable has
			// been set to true in the previous loop execution. If it is set to true,
			// it means that a break statement has been executed and thus the loop
			// must be terminated.

			// Keep track of the operations used to determine the old condition.
			llvm::SmallVector<mlir::Operation*, 3> oldConditionOps;

			for (auto& oldConditionOp : op.condition().getOps())
				oldConditionOps.push_back(&oldConditionOp);

			// Check the condition
			builder.setInsertionPointToStart(&op.condition().front());
			mlir::Value condition = builder.create<mlir::memref::LoadOp>(op->getLoc(), stopCondition());
			auto ifOp = builder.create<mlir::scf::IfOp>(op->getLoc(), builder.getI1Type(), condition, true);

			// If the break condition variable is set to true, return false from the
			// condition block in order to stop the loop execution.
			builder.setInsertionPointToStart(&ifOp.thenRegion().front());
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::scf::YieldOp>(op->getLoc(), falseValue);

			// Move the original condition check in the "else" branch
			for (auto& oldConditionOp : oldConditionOps)
				oldConditionOp->moveBefore(&ifOp.elseRegion().front(), ifOp.elseRegion().front().getOperations().end());

			auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
			builder.setInsertionPointAfter(conditionOp);
			mlir::Value oldCondition = typeConverter->materializeTargetConversion(builder, op->getLoc(), builder.getI1Type(), conditionOp.condition());
			builder.create<mlir::scf::YieldOp>(conditionOp->getLoc(), oldCondition);
			conditionOp->erase();

			// Return the new condition
			builder.setInsertionPointAfter(ifOp);
			mlir::Value newCondition = typeConverter->materializeSourceConversion(builder, op->getLoc(), BooleanType::get(op->getContext()), ifOp.getResult(0));
			builder.create<ConditionOp>(op->getLoc(), newCondition);
		}

		// Create the block of code to be executed if a break has not been called
		if (canReturn && !avoidableStatements.empty())
			wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

		return canReturn;
	}

	bool run(mlir::OpBuilder& builder, IfOp op, std::function<mlir::Value()> stopCondition)
	{
		bool canReturn = false;

		llvm::SmallVector<mlir::Region*, 3> regions;
		regions.push_back(&op.thenRegion());
		regions.push_back(&op.elseRegion());

		for (auto* region : regions)
		{
			// We need to make a copy of the operations list, or we would otherwise
			// invalidate the iterator as we change the IR structure.
			llvm::SmallVector<mlir::Operation*, 3> originalStatements;

			for (auto& statement : region->getOps())
				originalStatements.push_back(&statement);

			bool regionCanReturn = false;
			llvm::SmallVector<mlir::Operation*, 3> statements;
			llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

			for (auto& statement : originalStatements)
			{
				bool currentCanReturn = run(builder, statement, stopCondition);

				if (regionCanReturn)
					avoidableStatements.push_back(statement);
				else
					statements.push_back(statement);

				regionCanReturn |= currentCanReturn;
			}

			if (canReturn && !avoidableStatements.empty())
				wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

			canReturn |= regionCanReturn;
		}

		return canReturn;
	}

	bool run(mlir::OpBuilder& builder, WhileOp op, std::function<mlir::Value()> stopCondition)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);

		bool canReturn = false;
		llvm::SmallVector<mlir::Operation*, 3> statements;
		llvm::SmallVector<mlir::Operation*, 3> avoidableStatements;

		// We need to make a copy of the operations list, or we would otherwise
		// invalidate the iterator as we change the IR structure.
		llvm::SmallVector<mlir::Operation*, 3> originalStatements;

		for (auto& statement : op.body().getOps())
			originalStatements.push_back(&statement);

		for (auto& statement : originalStatements)
		{
			bool currentCanReturn = run(builder, statement, stopCondition);

			if (canReturn)
				avoidableStatements.push_back(statement);
			else
				statements.push_back(statement);

			canReturn |= currentCanReturn;
		}

		if (canReturn)
		{
			// The loop is supposed to be breakable. Thus, before checking the normal
			// condition, we first need to check if the break condition variable has
			// been set to true in the previous loop execution. If it is set to true,
			// it means that a break statement has been executed and thus the loop
			// must be terminated.

			// Keep track of the operations used to determine the old condition.
			llvm::SmallVector<mlir::Operation*, 3> oldConditionOps;

			for (auto& oldConditionOp : op.condition().getOps())
				oldConditionOps.push_back(&oldConditionOp);

			// Check the condition
			builder.setInsertionPointToStart(&op.condition().front());
			mlir::Value condition = builder.create<mlir::memref::LoadOp>(op->getLoc(), stopCondition());
			auto ifOp = builder.create<mlir::scf::IfOp>(op->getLoc(), builder.getI1Type(), condition, true);

			// If the break condition variable is set to true, return false from the
			// condition block in order to stop the loop execution.
			builder.setInsertionPointToStart(&ifOp.thenRegion().front());
			mlir::Value falseValue = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getBoolAttr(false));
			builder.create<mlir::scf::YieldOp>(op->getLoc(), falseValue);

			// Move the original condition check in the "else" branch
			for (auto& oldConditionOp : oldConditionOps)
				oldConditionOp->moveBefore(&ifOp.elseRegion().front(), ifOp.elseRegion().front().getOperations().end());

			auto conditionOp = mlir::cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
			builder.setInsertionPointAfter(conditionOp);
			mlir::Value oldCondition = typeConverter->materializeTargetConversion(builder, op->getLoc(), builder.getI1Type(), conditionOp.condition());
			builder.create<mlir::scf::YieldOp>(conditionOp->getLoc(), oldCondition);
			conditionOp->erase();

			// Return the new condition
			builder.setInsertionPointAfter(ifOp);
			mlir::Value newCondition = typeConverter->materializeSourceConversion(builder, op->getLoc(), BooleanType::get(op->getContext()), ifOp.getResult(0));
			builder.create<ConditionOp>(op->getLoc(), newCondition);
		}

		// Create the block of code to be executed if a break has not been called
		if (canReturn && !avoidableStatements.empty())
			wrapAvoidableOps(builder, op->getLoc(), avoidableStatements, stopCondition);

		return canReturn;
	}

	void wrapAvoidableOps(mlir::OpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Operation*> ops, std::function<mlir::Value()> stopCondition) const
	{
		if (ops.empty())
			return;

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(ops[0]);

		mlir::Value condition = builder.create<mlir::memref::LoadOp>(loc, stopCondition());
		mlir::Value falseValue = builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(false));
		condition = builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, condition, falseValue);
		auto ifOp = builder.create<mlir::scf::IfOp>(loc, condition);

		mlir::BlockAndValueMapping mapping;
		builder.setInsertionPointToStart(&ifOp.thenRegion().front());

		for (auto& op : ops)
		{
			builder.clone(*op, mapping);
			op->erase();
		}
	}

	mlir::TypeConverter* typeConverter;
};

class ControlFlowCanonicalizationPass : public mlir::PassWrapper<ControlFlowCanonicalizationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit ControlFlowCanonicalizationPass(unsigned int bitWidth)
			: bitWidth(bitWidth)
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::StandardOpsDialect>();
		registry.insert<mlir::scf::SCFDialect>();
		registry.insert<mlir::memref::MemRefDialect>();
	}

	void runOnOperation() override
	{
		if (auto status = removeBreakOperations(); failed(status))
		{
			mlir::emitError(getOperation().getLoc(), "Error in removing the break operations");
			return signalPassFailure();
		}

		if (auto status = removeEarlyReturnOperations(); failed(status))
		{
			mlir::emitError(getOperation().getLoc(), "Error in removing the early return operations");
			return signalPassFailure();
		}
	}

	private:
	mlir::LogicalResult removeBreakOperations()
	{
		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
		BreakRemover breakRemover(typeConverter);

		auto module = getOperation();
		mlir::OpBuilder builder(module);

		for (auto& op : module.getOps())
			if (op.hasTrait<mlir::OpTrait::FunctionLike>())
				if (auto status = breakRemover.run(builder, &op); failed(status))
					return status;

		return mlir::success();
	}

	mlir::LogicalResult removeEarlyReturnOperations()
	{
		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
		ReturnRemover returnRemover(typeConverter);

		auto module = getOperation();
		mlir::OpBuilder builder(module);

		for (auto& op : module.getOps())
			if (op.hasTrait<mlir::OpTrait::FunctionLike>())
				if (auto status = returnRemover.run(builder, &op); failed(status))
					return status;

		return mlir::success();
	}

	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createControlFlowCanonicalizationPass(unsigned int bitWidth)
{
	return std::make_unique<ControlFlowCanonicalizationPass>(bitWidth);
}
