#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
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

	class YieldOpLowering : public mlir::OpRewritePattern<YieldOp>
	{
		using mlir::OpRewritePattern<YieldOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(YieldOp op, mlir::PatternRewriter& rewriter) const final;
	};
}
