#include <modelica/mlirlowerer/ModelicaToStandard.hpp>

using namespace mlir;
using namespace modelica;
using namespace std;

void modelica::populateModelicaToStdConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
{
	patterns.insert<AddOpLowering>(context);
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

	mlir::Value result = rewriter.create<Op>(location, startValue, operands[i++]);

	for (; i < size; i++)
		result = rewriter.create<Op>(location, result, operands[i]);

	return result;
}

LogicalResult AddOpLowering::matchAndRewrite(AddOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	if (type.isSignlessInteger())
		rewriter.replaceOp(op, foldBinaryOp<mlir::AddIOp>(rewriter, location, operands));
	else if (type.isF16() || type.isF32() || type.isF64())
		rewriter.replaceOp(op, foldBinaryOp<mlir::AddFOp>(rewriter, location, operands));
	else
	{
		llvm::errs() << "Unsupported type\n";
		return failure();
	}

	return success();
}

LogicalResult SubOpLowering::matchAndRewrite(SubOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	if (type.isSignlessInteger())
		rewriter.replaceOp(op, foldBinaryOp<mlir::SubIOp>(rewriter, location, operands));
	else if (type.isF16() || type.isF32() || type.isF64())
		rewriter.replaceOp(op, foldBinaryOp<mlir::SubFOp>(rewriter, location, operands));
	else
	{
		llvm::errs() << "Unsupported type\n";
		return failure();
	}

	return success();
}

LogicalResult MulOpLowering::matchAndRewrite(MulOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	if (type.isSignlessInteger())
		rewriter.replaceOp(op, foldBinaryOp<mlir::MulIOp>(rewriter, location, operands));
	else if (type.isF16() || type.isF32() || type.isF64())
		rewriter.replaceOp(op, foldBinaryOp<mlir::MulFOp>(rewriter, location, operands));
	else
	{
		llvm::errs() << "Unsupported type\n";
		return failure();
	}

	return success();
}

LogicalResult DivOpLowering::matchAndRewrite(DivOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();
	mlir::Type type = op.getType();

	if (type.isSignlessInteger())
		rewriter.replaceOp(op, foldBinaryOp<mlir::UnsignedDivIOp>(rewriter, location, operands));
	else if (type.isF16() || type.isF32() || type.isF64())
		rewriter.replaceOp(op, foldBinaryOp<mlir::DivFOp>(rewriter, location, operands));
	else
	{
		llvm::errs() << "Unsupported type\n";
		return failure();
	}

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
