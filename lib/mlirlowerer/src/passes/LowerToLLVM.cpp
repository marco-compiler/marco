#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/LowerToLLVM.h>
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

LLVMLoweringPass::LLVMLoweringPass()
{
}

mlir::LogicalResult LLVMLoweringPass::stdToLLVMConversionPass(mlir::ModuleOp module)
{
	mlir::ConversionTarget target(getContext());
	target.addIllegalDialect<ModelicaDialect, mlir::StandardOpsDialect>();
	target.addIllegalOp<mlir::FuncOp>();

	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	target.addIllegalOp<mlir::LLVM::DialectCastOp>();
	target.addLegalOp<mlir::UnrealizedConversionCastOp>();
	target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

	mlir::LowerToLLVMOptions llvmLoweringOptions;
	llvmLoweringOptions.emitCWrappers = true;
	modelica::TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

	target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp>([&](mlir::Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
	target.addLegalOp<mlir::omp::TerminatorOp, mlir::omp::TaskyieldOp, mlir::omp::FlushOp, mlir::omp::BarrierOp, mlir::omp::TaskwaitOp>();

	mlir::OwningRewritePatternList patterns;
	mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
	mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

	return applyPartialConversion(module, target, std::move(patterns));
}

mlir::LogicalResult LLVMLoweringPass::castsFolderPass(mlir::ModuleOp module)
{
	mlir::ConversionTarget target(getContext());
	target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
	target.addLegalDialect<mlir::omp::OpenMPDialect>();
	target.addLegalDialect<mlir::LLVM::LLVMDialect>();
	target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

	mlir::OwningRewritePatternList patterns;
	patterns.insert<UnrealizedCastOpLowering>(&getContext());

	return applyFullConversion(module, target, std::move(patterns));
}

void LLVMLoweringPass::runOnOperation()
{
	auto module = getOperation();

	if (failed(stdToLLVMConversionPass(module)))
	{
		mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
		signalPassFailure();
		return;
	}

	module.dump();
	llvm::DebugFlag = true;

	if (failed(castsFolderPass(module)))
	{
		mlir::emitError(module.getLoc(), "Error in folding the casts operations\n");
		signalPassFailure();
	}
}

std::unique_ptr<mlir::Pass> modelica::createLLVMLoweringPass()
{
	return std::make_unique<LLVMLoweringPass>();
}
