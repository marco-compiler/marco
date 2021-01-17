#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ModelicaToStandard.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

LogicalResult MemCopyOpLowering::matchAndRewrite(MemCopyOp op, PatternRewriter& rewriter) const
{
	rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, op.source(), op.destination());
	return success();
}

LogicalResult NegateOpLowering::matchAndRewrite(NegateOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	mlir::Value operand = op->getOperand(0);
	mlir::Type type = op->getResultTypes()[0];

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		// There is no "negate" operation for integers in the Standard dialect
		rewriter.replaceOpWithNewOp<MulIOp>(op, rewriter.create<ConstantOp>(location, rewriter.getIntegerAttr(type, -1)), operand);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		rewriter.replaceOpWithNewOp<NegFOp>(op, operand);
		return success();
	}

	return failure();
}

LogicalResult AddOpLowering::matchAndRewrite(AddOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type =  op->getResultTypes()[0];

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	mlir::Value result = operands[0];

	if (type.isSignlessInteger())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<AddIOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<AddFOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	return failure();
}

LogicalResult SubOpLowering::matchAndRewrite(SubOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op->getResultTypes()[0];

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	mlir::Value result = operands[0];

	if (type.isSignlessInteger())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<SubIOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<SubFOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	return failure();
}

LogicalResult MulOpLowering::matchAndRewrite(MulOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op->getResultTypes()[0];

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	mlir::Value result = operands[0];

	if (type.isSignlessInteger())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<MulIOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<MulFOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	return failure();
}

LogicalResult DivOpLowering::matchAndRewrite(DivOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op->getResultTypes()[0];

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	mlir::Value result = operands[0];

	if (type.isSignlessInteger())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<SignedDivIOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		for (size_t i = 1; i < operands.size(); i++)
			result = rewriter.create<DivFOp>(location, result, operands[i]);

		rewriter.replaceOp(op, result);
		return success();
	}

	return failure();
}

LogicalResult EqOpLowering::matchAndRewrite(EqOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::eq, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OEQ, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult NotEqOpLowering::matchAndRewrite(NotEqOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ne, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::ONE, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult GtOpLowering::matchAndRewrite(GtOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sgt, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OGT, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult GteOpLowering::matchAndRewrite(GteOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sge, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OGE, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult LtOpLowering::matchAndRewrite(LtOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::slt, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OLT, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult LteOpLowering::matchAndRewrite(LteOp op, PatternRewriter& rewriter) const
{
	auto operands = op->getOperands();
	mlir::Type type = operands[0].getType();

	if (type.isa<VectorType>() || type.isa<TensorType>())
		type = type.cast<ShapedType>().getElementType();

	if (type.isSignlessInteger())
	{
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sle, operands[0], operands[1]);
		return success();
	}

	if (type.isF32() || type.isF64())
	{
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OLE, operands[0], operands[1]);
		return success();
	}

	return failure();
}

LogicalResult IfOpLowering::matchAndRewrite(IfOp op, PatternRewriter& rewriter) const
{
	auto loc = op.getLoc();

	// Split the current block
	auto* currentBlock = rewriter.getInsertionBlock();
	auto* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

	// "Then" region
	auto& thenRegion = op.thenRegion();
	auto* thenBlock = &thenRegion.front();
	auto thenYieldOp = dyn_cast<YieldOp>(thenRegion.back().getTerminator());

	// We need to check if it is effectively a YieldOp, because the block may
	// terminate with a break if the if is inside a loop.
	if (thenYieldOp)
	{
		rewriter.setInsertionPointToEnd(&thenRegion.back());
		rewriter.replaceOpWithNewOp<BranchOp>(thenYieldOp, continuation);
	}

	rewriter.inlineRegionBefore(thenRegion, continuation);

	// "Else" region
	auto &elseRegion = op.elseRegion();
	auto* elseBlock = continuation;

	if (!elseRegion.empty())
	{
		elseBlock = &elseRegion.front();
		auto elseYieldOp = dyn_cast<YieldOp>(elseRegion.back().getTerminator());

		if (elseYieldOp)
		{
			rewriter.setInsertionPointToEnd(&elseRegion.back());
			rewriter.replaceOpWithNewOp<BranchOp>(elseYieldOp, continuation);
		}

		rewriter.inlineRegionBefore(elseRegion, continuation);
	}

	// Branch selection
	rewriter.setInsertionPointToEnd(currentBlock);
	rewriter.create<CondBranchOp>(loc, op.condition(), thenBlock, elseBlock);

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
	Block* condition = &op.condition().front();
	Block* conditionLast = &op.condition().back();

	Block* body = &op.body().front();
	Block* bodyLast = &op.body().back();

	Block* step = &op.step().front();
	Block* stepLast = &op.step().back();

	Block* exit = &op.exit().front();

	rewriter.inlineRegionBefore(op.exit(), continuation);
	rewriter.inlineRegionBefore(op.step(), exit);
	rewriter.inlineRegionBefore(op.body(), step);
	rewriter.inlineRegionBefore(op.condition(), body);

	{
		// Start the for loop by branching to the "condition" region
		rewriter.setInsertionPointToEnd(currentBlock);
		rewriter.create<BranchOp>(location, condition, op.getOperands());
	}

	{
		// Branch to the loop body if the condition is satisfied
		rewriter.setInsertionPointToEnd(conditionLast);
		auto conditionOp = dyn_cast<ConditionOp>(conditionLast->getTerminator());

		llvm::SmallVector<mlir::Value, 3> args;

		for (size_t i = 1; i < conditionOp->getNumOperands(); i++)
			args.push_back(conditionOp.getOperand(i));

		rewriter.replaceOpWithNewOp<CondBranchOp>(conditionOp, conditionOp.getOperand(0), body, args, exit, ValueRange());
	}

	{
		// Replace "body" block terminator with a branch to the "step" block
		rewriter.setInsertionPointToEnd(bodyLast);
		auto yieldOp = dyn_cast<YieldOp>(bodyLast->getTerminator());

		// We need to check if it is effectively a YieldOp, because the body may
		// terminate with a break.
		if (yieldOp)
			rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, step, yieldOp->getOperands());
	}

	{
		// Branch to the condition check after incrementing the induction variable
		rewriter.setInsertionPointToEnd(stepLast);
		auto yieldOp = dyn_cast<YieldOp>(stepLast->getTerminator());
		rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, condition, yieldOp->getOperands());
	}

	{
		// Replace "exit" block terminator with a branch to the continuation block after the For operation
		rewriter.setInsertionPointToEnd(exit);
		auto yieldOp = cast<YieldOp>(exit->getTerminator());
		rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, continuation);
	}

	rewriter.eraseOp(op);
	return success();
}

LogicalResult WhileOpLowering::matchAndRewrite(WhileOp op, PatternRewriter& rewriter) const
{
	OpBuilder::InsertionGuard guard(rewriter);
	Location location = op.getLoc();

	// Split the current block
	Block *currentBlock = rewriter.getInsertionBlock();
	Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

	// Inline regions
	Block *exit = &op.exit().front();
	Block *body = &op.body().front();
	Block *bodyLast = &op.body().back();
	Block *condition = &op.condition().front();
	Block *conditionLast = &op.condition().back();

	rewriter.inlineRegionBefore(op.exit(), continuation);
	rewriter.inlineRegionBefore(op.body(), exit);
	rewriter.inlineRegionBefore(op.condition(), body);

	// Branch to the "condition" region
	rewriter.setInsertionPointToEnd(currentBlock);
	rewriter.create<BranchOp>(location, condition);

	// Replace "condition" block terminator with branch
	rewriter.setInsertionPointToEnd(conditionLast);
	auto condOp = cast<ConditionOp>(conditionLast->getTerminator());
	rewriter.replaceOpWithNewOp<CondBranchOp>(condOp, condOp.condition(), body, exit);

	// Replace "body" block terminator with branch
	rewriter.setInsertionPointToEnd(bodyLast);
	auto bodyYieldOp = dyn_cast<YieldOp>(bodyLast->getTerminator());

	// We need to check if it is effectively a YieldOp, because the body may
	// terminate with a break.
	if (bodyYieldOp)
		rewriter.replaceOpWithNewOp<BranchOp>(bodyYieldOp, condition);

	// Replace "exit" block terminator with branch
	rewriter.setInsertionPointToEnd(exit);
	auto exitYieldOp = cast<YieldOp>(exit->getTerminator());
	rewriter.replaceOpWithNewOp<BranchOp>(exitYieldOp, continuation);

	rewriter.eraseOp(op);
	return success();
}

LogicalResult YieldOpLowering::matchAndRewrite(YieldOp op, PatternRewriter& rewriter) const
{
	// The YieldOp is supposed to be converted by the parent loop lowerer,
	// which knows the structure of the control flow. In this sense, the
	// lowering process should not reach a point where it needs to invoke this
	// method.
	return failure();
}

LogicalResult ConditionOpLowering::matchAndRewrite(ConditionOp op, PatternRewriter& rewriter) const
{
	// The ConditionOp is supposed to be converted by the parent loop lowerer,
	// which knows the structure of the control flow. In this sense, the
	// lowering process should not reach a point where it needs to invoke this
	// method.
	return failure();
}

LogicalResult BreakOpLowering::matchAndRewrite(BreakOp op, PatternRewriter& rewriter) const
{
	rewriter.replaceOpWithNewOp<BranchOp>(op, op->getSuccessor(0));
	return success();
}

void ModelicaToStandardLoweringPass::runOnOperation()
{
	auto module = getOperation();
	ConversionTarget target(getContext());

	target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp>();
	target.addLegalDialect<StandardOpsDialect>();
	target.addLegalDialect<linalg::LinalgDialect>();

	// The Modelica dialect is defined as illegal, so that the conversion
	// will fail if any of its operations are not converted.
	target.addIllegalDialect<ModelicaDialect>();

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	populateModelicaToStdConversionPatterns(patterns, &getContext());

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyPartialConversion(module, target, move(patterns))))
		signalPassFailure();
}

void modelica::populateModelicaToStdConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
{
	// Generic operations
	patterns.insert<MemCopyOpLowering>(context);

	// Math operations
	patterns.insert<NegateOpLowering>(context);
	patterns.insert<AddOpLowering>(context);
	patterns.insert<SubOpLowering>(context);
	patterns.insert<MulOpLowering>(context);
	patterns.insert<DivOpLowering>(context);

	// Comparison operations
	patterns.insert<EqOpLowering>(context);
	patterns.insert<NotEqOpLowering>(context);
	patterns.insert<GtOpLowering>(context);
	patterns.insert<GteOpLowering>(context);
	patterns.insert<LtOpLowering>(context);
	patterns.insert<LteOpLowering>(context);

	// Control flow operations
	patterns.insert<IfOpLowering>(context);
	patterns.insert<ForOpLowering>(context);
	patterns.insert<WhileOpLowering>(context);
	patterns.insert<ConditionOpLowering>(context);
	patterns.insert<YieldOpLowering>(context);
	patterns.insert<BreakOpLowering>(context);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToStdPass()
{
	return std::make_unique<ModelicaToStandardLoweringPass>();
}
