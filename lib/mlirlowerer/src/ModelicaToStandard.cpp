#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/ModelicaToStandard.hpp>
#include <modelica/mlirlowerer/TypeConversion.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

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

//static pair<mlir::Value, mlir::Value> castToCommon(mlir::OpBuilder& builder, mlir::Value first, mlir::Value second) {
//	return std::make_pair<mlir::Value, mlir::Value>(nullptr, nullptr);
//}

LogicalResult NegateOpLowering::matchAndRewrite(NegateOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operand = op->getOperand(0);
	mlir::Type type = op.getType();

	mlir::Value result;

	if (operand.getType().isSignlessInteger())
		result = rewriter.create<MulIOp>(
				location,
				rewriter.create<ConstantOp>(location, rewriter.getI32IntegerAttr(-1)),
				operand);
	else if (operand.getType().isF32())
		result = rewriter.create<NegFOp>(location, operand);
	else
		assert(false && "Incompatible type");

	if (type.isSignlessInteger() && result.getType().isF32())
		result = rewriter.create<FPToSIOp>(location, result, type);
	else if (type.isF32() && result.getType().isSignlessInteger())
		result = rewriter.create<SIToFPOp>(location, result, type);

	rewriter.replaceOp(op, result);
	return success();
}

LogicalResult AddOpLowering::matchAndRewrite(AddOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		if (result.getType().isSignlessInteger() && operands[i].getType().isF32())
			result = rewriter.create<AddFOp>(
					location,
					rewriter.create<SIToFPOp>(location, result, operands[i].getType()),
					operands[i]);
		else if (result.getType().isF32() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<AddFOp>(
					location,
					result,
					rewriter.create<SIToFPOp>(location, operands[i], result.getType()));
		else if (result.getType().isF32() && operands[i].getType().isF32())
			result = rewriter.create<AddFOp>(location, result, operands[i]);
		else if (result.getType().isSignlessInteger() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<AddIOp>(location, result, operands[i]);
		else
			assert(false && "Incompatible types");
	}

	if (type.isSignlessInteger() && result.getType().isF32())
		result = rewriter.create<FPToSIOp>(location, result, type);
	else if (type.isF32() && result.getType().isSignlessInteger())
		result = rewriter.create<SIToFPOp>(location, result, type);

	rewriter.replaceOp(op, result);
	return success();
}

LogicalResult SubOpLowering::matchAndRewrite(SubOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		if (result.getType().isSignlessInteger() && operands[i].getType().isF32())
			result = rewriter.create<SubFOp>(
					location,
					rewriter.create<SIToFPOp>(location, result, operands[i].getType()),
					operands[i]);
		else if (result.getType().isF32() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<SubFOp>(
					location,
					result,
					rewriter.create<SIToFPOp>(location, operands[i], result.getType()));
		else if (result.getType().isF32() && operands[i].getType().isF32())
			result = rewriter.create<SubFOp>(location, result, operands[i]);
		else if (result.getType().isSignlessInteger() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<SubIOp>(location, result, operands[i]);
		else
			assert(false && "Incompatible types");
	}

	if (type.isSignlessInteger() && result.getType().isF32())
		result = rewriter.create<FPToSIOp>(location, result, type);
	else if (type.isF32() && result.getType().isSignlessInteger())
		result = rewriter.create<SIToFPOp>(location, result, type);

	rewriter.replaceOp(op, result);
	return success();
}

LogicalResult MulOpLowering::matchAndRewrite(MulOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		if (result.getType().isSignlessInteger() && operands[i].getType().isF32())
			result = rewriter.create<MulFOp>(
					location,
					rewriter.create<SIToFPOp>(location, result, operands[i].getType()),
					operands[i]);
		else if (result.getType().isF32() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<MulFOp>(
					location,
					result,
					rewriter.create<SIToFPOp>(location, operands[i], result.getType()));
		else if (result.getType().isF32() && operands[i].getType().isF32())
			result = rewriter.create<MulFOp>(location, result, operands[i]);
		else if (result.getType().isSignlessInteger() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<MulIOp>(location, result, operands[i]);
		else
			assert(false && "Incompatible types");
	}

	if (type.isSignlessInteger() && result.getType().isF32())
		result = rewriter.create<FPToSIOp>(location, result, type);
	else if (type.isF32() && result.getType().isSignlessInteger())
		result = rewriter.create<SIToFPOp>(location, result, type);

	rewriter.replaceOp(op, result);
	return success();
}

LogicalResult DivOpLowering::matchAndRewrite(DivOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		if (result.getType().isSignlessInteger() && operands[i].getType().isF32())
			result = rewriter.create<DivFOp>(
					location,
					rewriter.create<SIToFPOp>(location, result, operands[i].getType()),
					operands[i]);
		else if (result.getType().isF32() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<DivFOp>(
					location,
					result,
					rewriter.create<SIToFPOp>(location, operands[i], result.getType()));
		else if (result.getType().isF32() && operands[i].getType().isF32())
			result = rewriter.create<DivFOp>(location, result, operands[i]);
		else if (result.getType().isSignlessInteger() && operands[i].getType().isSignlessInteger())
			result = rewriter.create<SignedDivIOp>(location, result, operands[i]);
		else
			assert(false && "Incompatible types");
	}

	if (type.isSignlessInteger() && result.getType().isF32())
		result = rewriter.create<FPToSIOp>(location, result, type);
	else if (type.isF32() && result.getType().isSignlessInteger())
		result = rewriter.create<SIToFPOp>(location, result, type);

	rewriter.replaceOp(op, result);
	return success();
}

LogicalResult EqOpLowering::matchAndRewrite(EqOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::eq, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OEQ, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OEQ,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OEQ,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
}

LogicalResult NotEqOpLowering::matchAndRewrite(NotEqOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::ne, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::ONE, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::ONE,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::ONE,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
}

LogicalResult GtOpLowering::matchAndRewrite(GtOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sgt, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OGT, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OGT,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OGT,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
}

LogicalResult GteOpLowering::matchAndRewrite(GteOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sge, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OGE, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OGE,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OGE,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
}

LogicalResult LtOpLowering::matchAndRewrite(LtOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::slt, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OLT, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OLT,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OLT,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
}

LogicalResult LteOpLowering::matchAndRewrite(LteOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	if (operands[0].getType().isSignlessInteger() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpIOp>(op, CmpIPredicate::sle, operands[0], operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(op, CmpFPredicate::OLE, operands[0], operands[1]);
	else if (operands[0].getType().isSignlessInteger() && operands[1].getType().isF32())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OLE,
				rewriter.create<SIToFPOp>(location, operands[0], operands[1].getType()),
				operands[1]);
	else if (operands[0].getType().isF32() && operands[1].getType().isSignlessInteger())
		rewriter.replaceOpWithNewOp<CmpFOp>(
				op,
				CmpFPredicate::OLE,
				operands[0],
				rewriter.create<SIToFPOp>(location, operands[1], operands[0].getType()));
	else
		assert(false && "Incompatible types");

	return success();
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
