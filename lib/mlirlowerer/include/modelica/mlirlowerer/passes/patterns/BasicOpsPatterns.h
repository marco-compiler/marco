#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class CastOpLowering: public mlir::OpRewritePattern<CastOp>
	{
		using mlir::OpRewritePattern<CastOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(CastOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class CastCommonOpLowering: public mlir::OpRewritePattern<CastCommonOp>
	{
		using mlir::OpRewritePattern<CastCommonOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(CastCommonOp op, mlir::PatternRewriter& rewriter) const final;
	};

	class AssignmentOpLowering: public mlir::OpRewritePattern<AssignmentOp>
	{
		using mlir::OpRewritePattern<AssignmentOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const final;
	};
}
