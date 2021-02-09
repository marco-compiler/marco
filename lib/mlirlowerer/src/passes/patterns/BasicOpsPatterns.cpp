#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/patterns/BasicOpsPatterns.h>

using namespace mlir;
using namespace modelica;
using namespace std;

LogicalResult CastOpLowering::matchAndRewrite(CastOp op, PatternRewriter& rewriter) const
{
	auto location = op.getLoc();

	mlir::Value value = op.value();
	mlir::Type source = value.getType();
	mlir::Type destination = op->getResultTypes()[0];

	if (source == destination)
	{
		rewriter.replaceOp(op, value);
		return success();
	}

	mlir::Type sourceBase = source;
	mlir::Type destinationBase = destination;

	if (source.isa<ShapedType>())
	{
		sourceBase = source.cast<ShapedType>().getElementType();
		auto sourceShape = source.cast<ShapedType>().getShape();

		if (destination.isa<ShapedType>())
		{
			auto destinationShape = destination.cast<ShapedType>().getShape();
			destinationBase = destination.cast<ShapedType>().getElementType();
			assert(all_of(llvm::zip(sourceShape, destinationShape),
										[](const auto& pair)
										{
											return get<0>(pair) == get<1>(pair);
										}));

			destination = mlir::VectorType::get(destinationShape, destinationBase);
		}
		else
		{
			destination = mlir::VectorType::get(sourceShape, destinationBase);
		}
	}

	if (sourceBase == destinationBase)
	{
		rewriter.replaceOp(op, value);
		return success();
	}

	if (sourceBase.isSignlessInteger())
	{
		if (destinationBase.isa<FloatType>())
		{
			rewriter.replaceOpWithNewOp<SIToFPOp>(op, value, destination);
			return success();
		}

		if (destinationBase.isIndex())
		{
			rewriter.replaceOpWithNewOp<IndexCastOp>(op, value, destination);
			return success();
		}
	}

	if (sourceBase.isa<FloatType>())
	{
		if (destinationBase.isSignlessInteger())
		{
			rewriter.replaceOpWithNewOp<FPToSIOp>(op, value, destination);
			return success();
		}
	}

	if (sourceBase.isIndex())
	{
		if (destinationBase.isSignlessInteger())
		{
			rewriter.replaceOpWithNewOp<IndexCastOp>(op, value, destinationBase);
			return success();
		}

		if (destinationBase.isa<FloatType>())
		{
			mlir::Type integerType = rewriter.getIntegerType(sourceBase.getIntOrFloatBitWidth());
			mlir::Value integer = rewriter.create<IndexCastOp>(location, value, integerType);
			rewriter.replaceOpWithNewOp<SIToFPOp>(op, integer, destination);
			return success();
		}
	}

	return rewriter.notifyMatchFailure(op, "Unsupported type conversion");
}

LogicalResult CastCommonOpLowering::matchAndRewrite(CastCommonOp op, PatternRewriter& rewriter) const
{
	auto location = op->getLoc();
	SmallVector<mlir::Value, 3> values;

	for (auto tuple : llvm::zip(op->getOperands(), op->getResultTypes()))
	{
		mlir::Value castedValue = rewriter.create<CastOp>(location, get<0>(tuple), get<1>(tuple));
		values.push_back(castedValue);
	}

	rewriter.replaceOp(op, values);
	return success();
}

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
		rewriter.replaceOpWithNewOp<StoreOp>(op, source, op.destination());
	}

	return success();
}
