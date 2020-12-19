#include <modelica/mlirlowerer/ModelicaToStandard.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

void modelica::populateModelicaToStdConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
{
	patterns.insert<AddOpLowering>(context);
	patterns.insert<SubOpLowering>(context);
	patterns.insert<MulOpLowering>(context);
	patterns.insert<DivOpLowering>(context);
}

static mlir::Value cast(mlir::OpBuilder& builder, mlir::Value value, mlir::Type type)
{
	auto sourceType = value.getType();

	if (sourceType == type)
		return value;

	if (sourceType.isF16() || sourceType.isF32() || sourceType.isF64())
	{
		if (type.isSignedInteger() || type.isSignlessInteger())
			return builder.create<FPToSIOp>(builder.getUnknownLoc(), value, type);

		if (type.isUnsignedInteger())
			return builder.create<FPToUIOp>(builder.getUnknownLoc(), value, type);
	}
	else if (sourceType.isSignedInteger() || sourceType.isSignlessInteger())
	{
		if (type.isF16() || type.isF32() || type.isF64())
			return builder.create<SIToFPOp>(builder.getUnknownLoc(), value, type);
	}
	else if (sourceType.isUnsignedInteger())
	{
		if (type.isF16() || type.isF32() || type.isF64())
			return builder.create<UIToFPOp>(builder.getUnknownLoc(), value, type);
	}

	assert(false && "Can't cast value");
	return value;
}

template<typename Op>
static mlir::Value foldBinaryOp(PatternRewriter& rewriter, Location location, ValueRange operands, mlir::Value startValue = nullptr)
{
	size_t size = operands.size();

	if (startValue == nullptr)
		assert(size >= 2);
	else
		assert(size >= 1);

	size_t i = 0;

	if (startValue == nullptr)
	{
		startValue = operands[0];
		i = 1;
	}

	mlir::Value result;

	result = rewriter.create<Op>(location, startValue, operands[i++]);

	for (; i < size; i++)
		result = rewriter.create<Op>(location, result, operands[i]);

	return result;
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
