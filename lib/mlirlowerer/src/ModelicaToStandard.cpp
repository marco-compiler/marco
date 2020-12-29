#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ModelicaToStandard.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

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
		mlir::Value result = rewriter.create<MulIOp>(
				location,
				rewriter.create<ConstantOp>(location, rewriter.getIntegerAttr(type, -1)),
				operand);

		rewriter.replaceOp(op, result);
		return success();
	}

	if (type.isF64() || type.isF32())
	{
		mlir::Value result = rewriter.create<NegFOp>(location, operand);
		rewriter.replaceOp(op, result);
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

void ModelicaToStandardLoweringPass::runOnOperation()
{
	auto module = getOperation();
	ConversionTarget target(getContext());

	target.addLegalOp<ModuleOp, FuncOp>();
	target.addLegalDialect<scf::SCFDialect, StandardOpsDialect>();

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
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToStdPass()
{
	return std::make_unique<ModelicaToStandardLoweringPass>();
}
