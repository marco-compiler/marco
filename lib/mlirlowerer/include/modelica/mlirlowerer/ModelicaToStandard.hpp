#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/MathOps.hpp>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>

namespace modelica
{
	class AddOpLowering : public mlir::OpRewritePattern<AddOp>
	{
		using mlir::OpRewritePattern<AddOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(AddOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class SubOpLowering : public mlir::OpRewritePattern<SubOp>
	{
		using mlir::OpRewritePattern<SubOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class MulOpLowering : public mlir::OpRewritePattern<MulOp>
	{
		using mlir::OpRewritePattern<MulOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class DivOpLowering : public mlir::OpRewritePattern<DivOp>
	{
		using mlir::OpRewritePattern<DivOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(DivOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class ModelicaToStandardLoweringPass : public mlir::PassWrapper<ModelicaToStandardLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
		public:
		void runOnOperation() final;
	};

	/**
	 * Collect a set of patterns to lower from Modelica operations to operations
	 * within the Standard dialect.
	 *
	 * @param patterns patterns list to populate
	 * @param context	 MLIR context
	 */
	void populateModelicaToStdConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context);
}
