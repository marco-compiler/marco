#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <modelica/mlirlowerer/passes/LowerToLLVM.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen;

struct UnrealizedCastOpLowering : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>
{
	using mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op, mlir::PatternRewriter& rewriter) const override {
		rewriter.replaceOp(op, op->getOperands());
		return mlir::success();
	}
};

class LLVMLoweringPass : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	explicit LLVMLoweringPass(ModelicaToLLVMConversionOptions options)
			: options(std::move(options))
	{
	}

	mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module)
	{
		mlir::ConversionTarget target(getContext());
		target.addIllegalDialect<ModelicaDialect, mlir::StandardOpsDialect>();
		target.addIllegalOp<mlir::FuncOp>();

		target.addLegalDialect<mlir::LLVM::LLVMDialect>();
		target.addIllegalOp<mlir::LLVM::DialectCastOp>();
		target.addLegalOp<mlir::UnrealizedConversionCastOp>();
		target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();

		mlir::LowerToLLVMOptions llvmOptions;
		llvmOptions.emitCWrappers = options.emitCWrappers;

		modelica::codegen::TypeConverter typeConverter(&getContext(), llvmOptions);

		target.addDynamicallyLegalOp<mlir::omp::ParallelOp, mlir::omp::WsLoopOp>([&](mlir::Operation *op) { return typeConverter.isLegal(&op->getRegion(0)); });
		target.addLegalOp<mlir::omp::TerminatorOp, mlir::omp::TaskyieldOp, mlir::omp::FlushOp, mlir::omp::BarrierOp, mlir::omp::TaskwaitOp>();

		mlir::OwningRewritePatternList patterns;
		mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
		mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

		return applyPartialConversion(module, target, std::move(patterns));
	}

	mlir::LogicalResult castsFolderPass(mlir::ModuleOp module)
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

	void runOnOperation() override
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

	private:
	ModelicaToLLVMConversionOptions options;
};

std::unique_ptr<mlir::Pass> modelica::codegen::createLLVMLoweringPass(ModelicaToLLVMConversionOptions options)
{
	return std::make_unique<LLVMLoweringPass>(options);
}
