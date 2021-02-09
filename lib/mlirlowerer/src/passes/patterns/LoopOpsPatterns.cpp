#include <mlir/Dialect/SCF/SCF.h>
#include <modelica/mlirlowerer/passes/patterns/LoopOpsPatterns.h>

using namespace mlir;
using namespace modelica;
using namespace std;

LogicalResult IfOpLowering::matchAndRewrite(IfOp op, PatternRewriter& rewriter) const
{
	bool hasElseBlock = !op.elseRegion().empty();

	auto thenBuilder = [&](mlir::OpBuilder& builder, mlir::Location location)
	{
		rewriter.mergeBlocks(&op.thenRegion().front(), rewriter.getInsertionBlock());
	};

	if (hasElseBlock)
	{
		rewriter.create<scf::IfOp>(
				op.getLoc(), TypeRange(), op.condition(), thenBuilder,
				[&](mlir::OpBuilder& builder, mlir::Location location)
				{
					rewriter.mergeBlocks(&op.elseRegion().front(), builder.getInsertionBlock());
				});
	}
	else
	{
		rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(), op.condition(), thenBuilder, nullptr);
	}

	rewriter.eraseOp(op);
	return success();
}

LogicalResult ForOpLowering::matchAndRewrite(ForOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();

	// Split the current block
	Block* currentBlock = rewriter.getInsertionBlock();
	Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

	// Inline regions
	Block* conditionBlock = &op.condition().front();
	Block* bodyBlock = &op.body().front();
	Block* stepBlock = &op.step().front();

	rewriter.inlineRegionBefore(op.step(), continuation);
	rewriter.inlineRegionBefore(op.body(), stepBlock);
	rewriter.inlineRegionBefore(op.condition(), bodyBlock);

	// Start the for loop by branching to the "condition" region
	rewriter.setInsertionPointToEnd(currentBlock);
	rewriter.create<BranchOp>(location, conditionBlock, op.args());

	// The loop is supposed to be breakable. Thus, before checking the normal
	// condition, we first need to check if the break condition variable has
	// been set to true in the previous loop execution. If it is set to true,
	// it means that a break statement has been executed and thus the loop
	// must be terminated.

	rewriter.setInsertionPointToStart(conditionBlock);

	mlir::Value breakCondition = rewriter.create<LoadOp>(location, op.breakCondition());
	mlir::Value returnCondition = rewriter.create<LoadOp>(location, op.returnCondition());
	mlir::Value stopCondition = rewriter.create<OrOp>(location, breakCondition, returnCondition);

	mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
	mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

	auto ifOp = rewriter.create<scf::IfOp>(location, rewriter.getI1Type(), condition, true);
	Block* originalCondition = rewriter.splitBlock(conditionBlock, rewriter.getInsertionPoint());

	// If the break condition variable is set to true, return false from the
	// condition block in order to stop the loop execution.
	rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
	mlir::Value falseValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false));
	rewriter.create<scf::YieldOp>(location, falseValue);

	// Move the original condition check in the "else" branch
	rewriter.mergeBlocks(originalCondition, &ifOp.elseRegion().front());
	rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
	auto conditionOp = cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
	rewriter.replaceOpWithNewOp<scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

	// The original condition operation is converted to the SCF one and takes
	// as condition argument the result of the If operation, which is false
	// if a break must be executed or the intended condition value otherwise.
	rewriter.setInsertionPointAfter(ifOp);
	rewriter.create<CondBranchOp>(location, ifOp.getResult(0),
																bodyBlock, conditionOp.args(), continuation, ValueRange());

	// Replace "body" block terminator with a branch to the "step" block
	rewriter.setInsertionPointToEnd(bodyBlock);
	auto bodyYieldOp = cast<YieldOp>(bodyBlock->getTerminator());
	rewriter.replaceOpWithNewOp<BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp->getOperands());

	// Branch to the condition check after incrementing the induction variable
	rewriter.setInsertionPointToEnd(stepBlock);
	auto stepYieldOp = cast<YieldOp>(stepBlock->getTerminator());
	rewriter.replaceOpWithNewOp<BranchOp>(stepYieldOp, conditionBlock, stepYieldOp->getOperands());

	rewriter.eraseOp(op);
	return success();
}

LogicalResult WhileOpLowering::matchAndRewrite(WhileOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto whileOp = rewriter.create<scf::WhileOp>(location, TypeRange(), ValueRange());

	// The body block requires no modification apart from the change of the
	// terminator to the SCF dialect one.

	rewriter.createBlock(&whileOp.after());
	rewriter.mergeBlocks(&op.body().front(), &whileOp.after().front());
	mlir::Block* body = &whileOp.after().front();
	auto bodyTerminator = cast<YieldOp>(body->getTerminator());
	rewriter.setInsertionPointToEnd(body);
	rewriter.replaceOpWithNewOp<scf::YieldOp>(bodyTerminator, bodyTerminator.getOperands());

	// The loop is supposed to be breakable. Thus, before checking the normal
	// condition, we first need to check if the break condition variable has
	// been set to true in the previous loop execution. If it is set to true,
	// it means that a break statement has been executed and thus the loop
	// must be terminated.

	rewriter.createBlock(&whileOp.before());
	rewriter.setInsertionPointToStart(&whileOp.before().front());

	mlir::Value breakCondition = rewriter.create<LoadOp>(location, op.breakCondition());
	mlir::Value returnCondition = rewriter.create<LoadOp>(location, op.returnCondition());
	mlir::Value stopCondition = rewriter.create<OrOp>(location, breakCondition, returnCondition);

	mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
	mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

	auto ifOp = rewriter.create<scf::IfOp>(location, rewriter.getI1Type(), condition, true);

	// If the break condition variable is set to true, return false from the
	// condition block in order to stop the loop execution.
	rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
	mlir::Value falseValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false));
	rewriter.create<scf::YieldOp>(location, falseValue);

	// Move the original condition check in the "else" branch
	rewriter.mergeBlocks(&op.condition().front(), &ifOp.elseRegion().front());
	rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
	auto conditionOp = cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
	rewriter.replaceOpWithNewOp<scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

	// The original condition operation is converted to the SCF one and takes
	// as condition argument the result of the If operation, which is false
	// if a break must be executed or the intended condition value otherwise.
	rewriter.setInsertionPointAfter(ifOp);
	rewriter.create<scf::ConditionOp>(location, ifOp.getResult(0), conditionOp.args());

	rewriter.eraseOp(op);
	return success();
}

LogicalResult YieldOpLowering::matchAndRewrite(YieldOp op, PatternRewriter& rewriter) const
{
	rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
	return success();
}
