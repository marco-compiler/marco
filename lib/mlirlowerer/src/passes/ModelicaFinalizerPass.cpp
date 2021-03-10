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

struct UnrealizedCastOpLowering : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp>
{
	UnrealizedCastOpLowering(mlir::MLIRContext* ctx, mlir::TypeConverter& typeConverter)
			: mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp>(typeConverter, ctx, 1)
	{
	}

	mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp castOp, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const override {
		mlir::UnrealizedConversionCastOp::Adaptor transformed(operands);
		rewriter.replaceOp(castOp, transformed.inputs());
		return mlir::success();
	}
};

ModelicaFinalizerPass::ModelicaFinalizerPass()
{
}

void ModelicaFinalizerPass::runOnOperation()
{
	auto module = getOperation();

	mlir::ConversionTarget target(getContext());
	target.addIllegalDialect<ModelicaDialect>();
	target.addIllegalDialect<mlir::StandardOpsDialect>();
	target.addIllegalOp<mlir::FuncOp>();

	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
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
	populateModelicaFinalizerPatterns(patterns, &getContext(), typeConverter);

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyPartialConversion(module, target, std::move(patterns))))
	{
		mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
		signalPassFailure();
	}
}

void modelica::populateModelicaFinalizerPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, mlir::TypeConverter& typeConverter)
{
	//patterns.insert<TestOpLowering>(context, typeConverter);
	//patterns.insert<UnrealizedCastOpLowering>(context, typeConverter);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaFinalizerPass()
{
	return std::make_unique<ModelicaFinalizerPass>();
}
