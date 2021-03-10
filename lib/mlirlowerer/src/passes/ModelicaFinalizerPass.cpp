#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ModelicaFinalizerPass.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>

using namespace modelica;

struct UnrealizedCastOpLowering : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>
{
	using mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, mlir::PatternRewriter& rewriter) const override {
		rewriter.replaceOp(op, op->getOperands());
		return mlir::success();
	}
};

ModelicaFinalizerPass::ModelicaFinalizerPass()
{
}

mlir::LogicalResult ModelicaFinalizerPass::stdToLLVMConversionPass(mlir::ModuleOp module)
{
	mlir::ConversionTarget target(getContext());
	target.addIllegalDialect<ModelicaDialect>();
	target.addIllegalDialect<mlir::StandardOpsDialect>();
	target.addIllegalOp<mlir::FuncOp>();

	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	target.addIllegalOp<mlir::LLVM::DialectCastOp>();
	target.addLegalOp<mlir::UnrealizedConversionCastOp>();
	target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

	// Create the type converter. We also need to create
	// the functions wrapper, in order to JIT it easily.
	mlir::LowerToLLVMOptions llvmLoweringOptions;
	llvmLoweringOptions.emitCWrappers = true;
	modelica::TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	populateStdToLLVMConversionPatterns(typeConverter, patterns);

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	return applyPartialConversion(module, target, std::move(patterns));
}

mlir::LogicalResult ModelicaFinalizerPass::castsFolderPass(mlir::ModuleOp module)
{
	mlir::ConversionTarget target(getContext());
	target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

	mlir::OwningRewritePatternList patterns;
	patterns.insert<UnrealizedCastOpLowering>(&getContext());

	return applyFullConversion(module, target, std::move(patterns));
}

void ModelicaFinalizerPass::runOnOperation()
{
	auto module = getOperation();

	if (failed(stdToLLVMConversionPass(module)))
	{
		mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
		signalPassFailure();
		return;
	}

	if (failed(castsFolderPass(module)))
	{
		mlir::emitError(module.getLoc(), "Error in folding the casts operations\n");
		signalPassFailure();
	}
}

std::unique_ptr<mlir::Pass> modelica::createModelicaFinalizerPass()
{
	return std::make_unique<ModelicaFinalizerPass>();
}
