#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/LowerToCFG.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <stack>

using namespace modelica::codegen;

class CFGLowerer
{
	public:
	CFGLowerer(mlir::TypeConverter& typeConverter) : typeConverter(&typeConverter)
	{
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, mlir::Operation* function)
	{
		assert(function->hasTrait<mlir::OpTrait::FunctionLike>());
		auto& body = mlir::impl::getFunctionBody(function);

		if (body.empty())
			return mlir::success();

		llvm::SmallVector<mlir::Operation*, 3> bodyOps;

		for (auto& bodyOp : body.getOps())
			bodyOps.push_back(&bodyOp);

		mlir::Block* functionReturnBlock = &body.back();

		for (auto& bodyOp : bodyOps)
			if (auto status = run(builder, bodyOp, nullptr, functionReturnBlock); failed(status))
				return status;

		// Remove the unreachable blocks
		std::stack<mlir::Block*> unreachableBlocks;

		for (auto& block : body.getBlocks())
			if (block.hasNoPredecessors() && !block.isEntryBlock())
				unreachableBlocks.push(&block);

		do {
			while (!unreachableBlocks.empty())
			{
				unreachableBlocks.top()->erase();
				unreachableBlocks.pop();
			}

			for (auto& block : body.getBlocks())
				if (block.hasNoPredecessors() && !block.isEntryBlock())
					unreachableBlocks.push(&block);
		} while (!unreachableBlocks.empty());

		return mlir::success();
	}

	private:
	mlir::LogicalResult run(mlir::OpBuilder& builder, mlir::Operation* op, mlir::Block* loopExitBlock, mlir::Block* functionReturnBlock)
	{
		if (auto breakOp = mlir::dyn_cast<BreakOp>(op))
			return run(builder, breakOp, loopExitBlock);

		if (auto forOp = mlir::dyn_cast<ForOp>(op))
			return run(builder, forOp, functionReturnBlock);

		if (auto ifOp = mlir::dyn_cast<IfOp>(op))
			return run(builder, ifOp, loopExitBlock, functionReturnBlock);

		if (auto whileOp = mlir::dyn_cast<WhileOp>(op))
			return run(builder, whileOp, functionReturnBlock);

		if (auto returnOp = mlir::dyn_cast<ReturnOp>(op))
			return run(builder, returnOp, functionReturnBlock);

		return mlir::success();
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, BreakOp op, mlir::Block* loopExitBlock)
	{
		if (loopExitBlock == nullptr)
			return mlir::failure();

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		mlir::Block* currentBlock = builder.getInsertionBlock();
		mlir::Block* continuation = currentBlock->splitBlock(op);

		builder.setInsertionPointToEnd(currentBlock);
		builder.create<mlir::BranchOp>(op->getLoc(), loopExitBlock);

		op->erase();
		return mlir::success();
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, ForOp op, mlir::Block* functionReturnBlock)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		// Split the current block
		mlir::Block* currentBlock = builder.getInsertionBlock();
		mlir::Block* continuation = currentBlock->splitBlock(op);

		// Keep the references to the op blocks
		mlir::Block* conditionFirst = &op.condition().front();
		mlir::Block* conditionLast = &op.condition().back();
		mlir::Block* bodyFirst = &op.body().front();
		mlir::Block* bodyLast = &op.body().back();
		mlir::Block* stepFirst = &op.step().front();
		mlir::Block* stepLast = &op.step().back();

		// Inline the regions
		inlineRegionBefore(op.condition(), continuation);
		inlineRegionBefore(op.body(), continuation);
		inlineRegionBefore(op.step(), continuation);

		// Start the for loop by branching to the "condition" region
		builder.setInsertionPointToEnd(currentBlock);
		builder.create<mlir::BranchOp>(op->getLoc(), conditionFirst, op.args());

		// Check the condition
		auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());
		builder.setInsertionPoint(conditionOp);

		mlir::Value conditionValue = typeConverter->materializeTargetConversion(
				builder, conditionOp.condition().getLoc(), builder.getI1Type(), conditionOp.condition());

		builder.create<mlir::CondBranchOp>(
				conditionOp->getLoc(), conditionValue, bodyFirst, conditionOp.args(), continuation, llvm::None);

		conditionOp->erase();

		// If present, replace "body" block terminator with a branch to the
		// "step" block. If not present, just place the branch.
		builder.setInsertionPointToEnd(bodyLast);
		llvm::SmallVector<mlir::Value, 3> bodyYieldValues;

		if (auto yieldOp = mlir::dyn_cast<YieldOp>(bodyLast->getTerminator()))
		{
			for (mlir::Value value : yieldOp.values())
				bodyYieldValues.push_back(value);

			yieldOp->erase();
		}

		builder.create<mlir::BranchOp>(op->getLoc(), stepFirst, bodyYieldValues);

		// Branch to the condition check after incrementing the induction variable
		builder.setInsertionPointToEnd(stepLast);
		llvm::SmallVector<mlir::Value, 3> stepYieldValues;

		if (auto yieldOp = mlir::dyn_cast<YieldOp>(stepLast->getTerminator()))
		{
			for (mlir::Value value : yieldOp.values())
				stepYieldValues.push_back(value);

			yieldOp->erase();
		}

		builder.create<mlir::BranchOp>(op->getLoc(), conditionFirst, stepYieldValues);

		// Erase the operation
		op->erase();

		// Recurse on the body operations
		return recurse(builder, bodyFirst, bodyLast, continuation, functionReturnBlock);
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, IfOp op, mlir::Block* loopExitBlock, mlir::Block* functionReturnBlock)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		// Split the current block
		mlir::Block* currentBlock = builder.getInsertionBlock();
		mlir::Block* continuation = currentBlock->splitBlock(op);

		// Keep the references to the op blocks
		mlir::Block* thenFirst = &op.thenRegion().front();
		mlir::Block* thenLast = &op.thenRegion().back();

		// Inline the regions
		inlineRegionBefore(op.thenRegion(), continuation);
		builder.setInsertionPointToEnd(currentBlock);

		mlir::Value conditionValue = typeConverter->materializeTargetConversion(
				builder, op.condition().getLoc(), builder.getI1Type(), op.condition());

		if (op.elseRegion().empty())
		{
			// Branch to the "then" region or to the continuation block according
			// to the condition.

			builder.create<mlir::CondBranchOp>(
					op->getLoc(), conditionValue, thenFirst, llvm::None, continuation, llvm::None);

			builder.setInsertionPointToEnd(thenLast);
			builder.create<mlir::BranchOp>(op->getLoc(), continuation);

			// Erase the operation
			op->erase();

			// Recurse on the body operations
			if (auto status = recurse(builder, thenFirst, thenLast, loopExitBlock, functionReturnBlock);
					failed(status))
				return status;
		}
		else
		{
			// Branch to the "then" region or to the "else" region according
			// to the condition.
			mlir::Block* elseFirst = &op.elseRegion().front();
			mlir::Block* elseLast = &op.elseRegion().back();

			inlineRegionBefore(op.elseRegion(), continuation);

			builder.create<mlir::CondBranchOp>(
					op->getLoc(), conditionValue, thenFirst, llvm::None, elseFirst, llvm::None);

			// Branch to the continuation block
			builder.setInsertionPointToEnd(thenLast);
			builder.create<mlir::BranchOp>(op->getLoc(), continuation);

			builder.setInsertionPointToEnd(elseLast);
			builder.create<mlir::BranchOp>(op->getLoc(), continuation);

			// Erase the operation
			op->erase();

			if (auto status = recurse(builder, elseFirst, elseLast, loopExitBlock, functionReturnBlock);
					failed(status))
				return status;
		}

		return mlir::success();
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, WhileOp op, mlir::Block* functionReturnBlock)
	{
		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		// Split the current block
		mlir::Block* currentBlock = builder.getInsertionBlock();
		mlir::Block* continuation = currentBlock->splitBlock(op);

		// Keep the references to the op blocks
		mlir::Block* conditionFirst = &op.condition().front();
		mlir::Block* conditionLast = &op.condition().back();

		mlir::Block* bodyFirst = &op.body().front();
		mlir::Block* bodyLast = &op.body().back();

		// Inline the regions
		inlineRegionBefore(op.condition(), continuation);
		inlineRegionBefore(op.body(), continuation);

		// Branch to the "condition" region
		builder.setInsertionPointToEnd(currentBlock);
		builder.create<mlir::BranchOp>(op->getLoc(), conditionFirst);

		// Branch to the "body" region
		builder.setInsertionPointToEnd(conditionLast);
		auto conditionOp = mlir::cast<ConditionOp>(conditionLast->getTerminator());

		mlir::Value conditionValue = typeConverter->materializeTargetConversion(
				builder, conditionOp->getLoc(), builder.getI1Type(), conditionOp.condition());

		builder.create<mlir::CondBranchOp>(
				op->getLoc(), conditionValue, bodyFirst, llvm::None, continuation, llvm::None);

		conditionOp->erase();

		// Branch back to the "condition" region
		builder.setInsertionPointToEnd(bodyLast);
		builder.create<mlir::BranchOp>(op->getLoc(), conditionFirst);

		// Erase the operation
		op->erase();

		// Recurse on the body operations
		return recurse(builder, bodyFirst, bodyLast, continuation, functionReturnBlock);
	}

	mlir::LogicalResult run(mlir::OpBuilder& builder, ReturnOp op, mlir::Block* functionReturnBlock)
	{
		if (functionReturnBlock == nullptr)
			return mlir::failure();

		mlir::OpBuilder::InsertionGuard guard(builder);
		builder.setInsertionPoint(op);

		mlir::Block* currentBlock = builder.getInsertionBlock();
		mlir::Block* continuation = currentBlock->splitBlock(op);

		builder.setInsertionPointToEnd(currentBlock);
		builder.create<mlir::BranchOp>(op->getLoc(), functionReturnBlock);

		op->erase();
		return mlir::success();
	}

	void inlineRegionBefore(mlir::Region& region, mlir::Block* before)
	{
		before->getParent()->getBlocks().splice(before->getIterator(), region.getBlocks());
	}

	mlir::LogicalResult recurse(mlir::OpBuilder& builder, mlir::Block* first, mlir::Block* last, mlir::Block* loopExitBlock, mlir::Block* functionReturnBlock)
	{
		llvm::SmallVector<mlir::Operation*, 3> ops;
		auto it = first->getIterator();

		do {
			for (auto& op : it->getOperations())
				ops.push_back(&op);
		} while (it++ != last->getIterator());

		for (auto& op : ops)
			if (auto status = run(builder, op, loopExitBlock, functionReturnBlock); failed(status))
				return status;

		return mlir::success();
	}

	mlir::TypeConverter* typeConverter;
};

class LowerToCFGPass : public mlir::PassWrapper<LowerToCFGPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit LowerToCFGPass(unsigned int bitWidth)
			: bitWidth(bitWidth)
	{
	}

	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::StandardOpsDialect>();
	}

	void runOnOperation() override
	{
		if (failed(convertModelicaLoops()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in converting the Modelica control flow operations");
			return signalPassFailure();
		}

		if (failed(convertSCF()))
		{
			mlir::emitError(getOperation().getLoc(), "Error in converting the SCF ops");
			return signalPassFailure();
		}
	}

	mlir::LogicalResult convertModelicaLoops()
	{
		auto module = getOperation();
		mlir::OpBuilder builder(module);

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);
		CFGLowerer lowerer(typeConverter);

		llvm::SmallVector<mlir::Operation*, 3> functions;

		module->walk([&functions](mlir::Operation* op) {
			if (op->hasTrait<mlir::OpTrait::FunctionLike>())
				functions.push_back(op);
		});

		for (auto function : functions)
			if (auto status = lowerer.run(builder, function); failed(status))
				return status;

		return mlir::success();
	}

	mlir::LogicalResult convertSCF()
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());

		target.addIllegalOp<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::ParallelOp, mlir::scf::WhileOp>();
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		mlir::LowerToLLVMOptions llvmLoweringOptions(&getContext());
		TypeConverter typeConverter(&getContext(), llvmLoweringOptions, bitWidth);

		mlir::OwningRewritePatternList patterns(&getContext());
		mlir::populateLoopToStdConversionPatterns(patterns);

		return applyPartialConversion(module, target, std::move(patterns));
	}

	unsigned int bitWidth;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createLowerToCFGPass(unsigned int bitWidth)
{
	return std::make_unique<LowerToCFGPass>(bitWidth);
}
