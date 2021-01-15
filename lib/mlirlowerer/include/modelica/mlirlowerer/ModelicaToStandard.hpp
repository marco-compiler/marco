#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.hpp>

namespace modelica
{
	class ArrayCopyOpLowering : public mlir::OpRewritePattern<ArrayCopyOp>
	{
		using mlir::OpRewritePattern<ArrayCopyOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(ArrayCopyOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class NegateOpLowering : public mlir::OpRewritePattern<NegateOp>
	{
		using mlir::OpRewritePattern<NegateOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(NegateOp op, mlir::PatternRewriter& rewriter) const final;
	};

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

	class EqOpLowering : public mlir::OpRewritePattern<EqOp>
	{
		using mlir::OpRewritePattern<EqOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(EqOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class NotEqOpLowering : public mlir::OpRewritePattern<NotEqOp>
	{
		using mlir::OpRewritePattern<NotEqOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(NotEqOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class GtOpLowering : public mlir::OpRewritePattern<GtOp>
	{
		using mlir::OpRewritePattern<GtOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(GtOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class GteOpLowering : public mlir::OpRewritePattern<GteOp>
	{
		using mlir::OpRewritePattern<GteOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(GteOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class LtOpLowering : public mlir::OpRewritePattern<LtOp>
	{
		using mlir::OpRewritePattern<LtOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(LtOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class LteOpLowering : public mlir::OpRewritePattern<LteOp>
	{
		using mlir::OpRewritePattern<LteOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(LteOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class IfOpLowering : public mlir::OpRewritePattern<IfOp>
	{
		using mlir::OpRewritePattern<IfOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(IfOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class ForOpLowering : public mlir::OpRewritePattern<ForOp>
	{
		using mlir::OpRewritePattern<ForOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(ForOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class WhileOpLowering : public mlir::OpRewritePattern<WhileOp>
	{
		using mlir::OpRewritePattern<WhileOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(WhileOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class ConditionOpLowering : public mlir::OpRewritePattern<ConditionOp>
	{
		using mlir::OpRewritePattern<ConditionOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(ConditionOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class YieldOpLowering : public mlir::OpRewritePattern<YieldOp>
	{
		using mlir::OpRewritePattern<YieldOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(YieldOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class BreakOpLowering : public mlir::OpRewritePattern<BreakOp>
	{
		using mlir::OpRewritePattern<BreakOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(BreakOp op, mlir::PatternRewriter& rewriter) const final;
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

	/**
	 * Create a pass to convert Modelica operations to Standard ones.
	 *
	 * @return pass
	 */
	std::unique_ptr<mlir::Pass> createModelicaToStdPass();
}
