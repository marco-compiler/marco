#pragma once

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>

namespace modelica
{
	class AssignmentOpLowering: public mlir::OpRewritePattern<AssignmentOp>
	{
		using mlir::OpRewritePattern<AssignmentOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(AssignmentOp op, mlir::PatternRewriter& rewriter) const final;
	};
}
