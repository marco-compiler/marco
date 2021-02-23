#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/patterns/BasicOpsPatterns.h>

using namespace mlir;
using namespace modelica;
using namespace std;

LogicalResult AssignmentOpLowering::matchAndRewrite(AssignmentOp op, PatternRewriter& rewriter) const
{
	mlir::Value source = op.source();
	mlir::Type sourceType = source.getType();

	if (sourceType.isa<MemRefType>())
	{
		rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, source, op.destination());
	}
	else if (sourceType.isa<VectorType>())
	{
		mlir::Value zeroValue = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
		SmallVector<mlir::Value, 3> indexes(sourceType.cast<ShapedType>().getRank(), zeroValue);
		rewriter.replaceOpWithNewOp<AffineVectorStoreOp>(op, source, op.destination(), indexes);
	}
	else
	{
		rewriter.replaceOpWithNewOp<mlir::StoreOp>(op, source, op.destination());
	}

	return success();
}
