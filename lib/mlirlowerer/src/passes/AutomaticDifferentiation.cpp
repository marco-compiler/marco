#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/AutomaticDifferentiation.h>

using namespace modelica::codegen;

struct DerFunctionOpPattern : public mlir::OpRewritePattern<DerFunctionOp>
{
	using mlir::OpRewritePattern<DerFunctionOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(DerFunctionOp op, mlir::PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();

		auto module = op->getParentOfType<mlir::ModuleOp>();
		auto base = module.lookupSymbol<FunctionOp>(op.derivedFunction());
		auto returnOp = mlir::cast<ReturnOp>(base.getBody().back().getTerminator());

		llvm::SmallVector<llvm::StringRef, 3> argsNames;
		llvm::SmallVector<llvm::StringRef, 3> resultsNames;

		for (const auto& argName : base.argsNames())
			argsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

		for (const auto& argName : base.resultsNames())
			resultsNames.push_back(argName.cast<mlir::StringAttr>().getValue());

		auto function = rewriter.replaceOpWithNewOp<FunctionOp>(op, op.getName(), base.getType(), argsNames, resultsNames);

		mlir::BlockAndValueMapping mapping;

		// Clone the blocks structure of the function to be derived. The
		// operations contained in the blocks are not copied.

		for (auto& block : base.getRegion().getBlocks())
		{
			mlir::Block* clonedBlock = rewriter.createBlock(
					&function.getRegion(), function.getRegion().end(), block.getArgumentTypes());

			mapping.map(&block, clonedBlock);
		}

		// Iterate over the original operations and create their derivatives
		// (if possible) inside the new function.

		for (auto& baseBlock : llvm::enumerate(base.getBody().getBlocks()))
		{
			auto& block = *std::next(function.getBlocks().begin(), baseBlock.index());
			rewriter.setInsertionPointToStart(&block);

			// Map the original block arguments to the new block ones
			//for (const auto& [original, mapped] : llvm::zip(baseBlock.value().getArguments(), block->getArguments()))
		//		mapping.map(original, mapped);

			rewriter.mergeBlocks(&baseBlock.value(), &block, block.getArguments());
			//for (auto& baseOp : baseBlock.value().getOperations())
			//	rewriter.clone(baseOp, mapping);
		}

		return mlir::success();
	}
};

static void populateAutomaticDifferentiationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context)
{
	patterns.insert<DerFunctionOpPattern>(context);
}

class AutomaticDifferentiationPass: public mlir::PassWrapper<AutomaticDifferentiationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<ModelicaDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		mlir::ConversionTarget target(getContext());
		target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) { return true; });

		target.addDynamicallyLegalOp<DerFunctionOp>([](DerFunctionOp op) {
			// Mark the operation as illegal only if the function to be derived
			// is a standard one. This way, in a chain of partial derivatives one
			// derivation will take place only when all the previous one have
			// been computed.

			auto module = op->getParentOfType<mlir::ModuleOp>();
			auto* derivedFunction = module.lookupSymbol(op.derivedFunction());
			return !mlir::isa<FunctionOp>(derivedFunction);
		});

		mlir::OwningRewritePatternList patterns(&getContext());
		populateAutomaticDifferentiationPatterns(patterns, &getContext());

		if (failed(applyPartialConversion(module, target, std::move(patterns))))
		{
			mlir::emitError(module.getLoc(), "Error during automatic differentiation\n");
			signalPassFailure();
		}
	}
};

std::unique_ptr<mlir::Pass> modelica::codegen::createAutomaticDifferentiationPass()
{
	return std::make_unique<AutomaticDifferentiationPass>();
}
