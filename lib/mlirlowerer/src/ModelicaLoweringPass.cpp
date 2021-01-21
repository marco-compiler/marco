#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ModelicaLoweringPass.hpp>

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
	bool hasElseBlock = !op.elseRegion().empty();

	auto thenBuilder = [&](mlir::OpBuilder& builder, mlir::Location location)
	{
		rewriter.mergeBlocks(&op.thenRegion().front(), rewriter.getInsertionBlock());
	};

	if (hasElseBlock)
	{
		auto ifOp = rewriter.create<scf::IfOp>(
				op.getLoc(), TypeRange(), op.condition(), thenBuilder,
				[&](mlir::OpBuilder& builder, mlir::Location location)
				{
					rewriter.mergeBlocks(&op.elseRegion().front(), builder.getInsertionBlock());
				});
		ifOp.dump();
	}
	else
	{
		auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(), op.condition(), thenBuilder, nullptr);
		ifOp.dump();
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

void ModelicaLoweringPass::runOnOperation()
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
	populateModelicaConversionPatterns(patterns, &getContext());
	populateLoopToStdConversionPatterns(patterns, &getContext());

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyFullConversion(module, target, move(patterns))))
		signalPassFailure();
}

void modelica::populateModelicaConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
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
	patterns.insert<YieldOpLowering>(context);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaLoweringPass()
{
	return std::make_unique<ModelicaLoweringPass>();
}
