#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class MulOpLowering : public mlir::OpRewritePattern<MulOp>
	{
		using mlir::OpRewritePattern<MulOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(MulOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class CrossProductOpLowering : public mlir::OpRewritePattern<CrossProductOp>
	{
		using mlir::OpRewritePattern<CrossProductOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(CrossProductOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class DivOpLowering : public mlir::OpRewritePattern<DivOp>
	{
		using mlir::OpRewritePattern<DivOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(DivOp op, mlir::PatternRewriter& rewriter) const final;
	};
}
