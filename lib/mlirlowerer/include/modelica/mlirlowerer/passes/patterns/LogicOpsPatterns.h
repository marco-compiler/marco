#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class NegateOpLowering : public mlir::OpRewritePattern<NegateOp>
	{
		using mlir::OpRewritePattern<NegateOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(NegateOp op, mlir::PatternRewriter& rewriter) const final;
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
}
