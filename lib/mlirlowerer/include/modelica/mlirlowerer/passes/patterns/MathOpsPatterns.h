#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class CrossProductOpLowering : public mlir::OpRewritePattern<CrossProductOp>
	{
		using mlir::OpRewritePattern<CrossProductOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(CrossProductOp op, mlir::PatternRewriter& rewriter) const final;
	};
}
