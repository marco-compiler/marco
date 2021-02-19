#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/patterns/MathOpsPatterns.h>

using namespace mlir;
using namespace modelica;
using namespace std;

/**
 * Get the base type that composes a shaped type.
 * Example: memref<3xi32> has base type i32.
 */
mlir::Type getBaseType(mlir::Type type)
{
	while (type.isa<ShapedType>())
		type = type.cast<ShapedType>().getElementType();

	return type;
}

/**
 * Load the data from a memref into a vector.
 *
 * @param builder operation builder
 * @param value   memref value
 * @return vector value
 */
mlir::Value memRefToVector(mlir::OpBuilder* builder, mlir::Value value)
{
	auto memRefType = value.getType().cast<MemRefType>();
	mlir::VectorType vectorType = mlir::VectorType::get(memRefType.getShape(), memRefType.getElementType());
	mlir::Value zeroValue = builder->create<ConstantOp>(value.getLoc(), builder->getIndexAttr(0));
	SmallVector<mlir::Value, 3> indexes(memRefType.getRank(), zeroValue);
	return builder->create<AffineVectorLoadOp>(value.getLoc(), vectorType, value, indexes);
}

/**
 * Get the most generic base type.
 *
 * The comparison is done considering that:
 *   - float is more general than integer
 *   - integer is more general than index
 *
 * @param x first type
 * @param y second type
 * @return most general base type
 */
mlir::Type getMostGeneralBaseType(mlir::Type x, mlir::Type y)
{
	x = getBaseType(x);
	y = getBaseType(y);

	if (x.isIndex())
		return y;

	if (x.isa<mlir::IntegerType>())
	{
		if (y.isa<FloatType>())
			return y;

		return x;
	}

	if (x.isa<mlir::FloatType>())
		if (y.isa<FloatType>() && y.getIntOrFloatBitWidth() > x.getIntOrFloatBitWidth())
			return y;

	return x;
}

LogicalResult AddOpLowering::matchAndRewrite(AddOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	mlir::Type resultType = op->getResultTypes()[0];
	mlir::Value result = op.getOperand(0);

	// The operation takes at least two operands. If more than two are provided,
	// they are summed in couples.

	for (size_t i = 1; i < op.getNumOperands(); i++)
	{
		mlir::Value x = result;
		mlir::Value y = op.getOperand(i);

		if (x.getType().isa<ShapedType>())
		{
			assert(y.getType().isa<ShapedType>());
			auto xShapedType = x.getType().cast<ShapedType>();
			auto yShapedType = y.getType().cast<ShapedType>();

			// Create the result buffer
			mlir::Type resultBaseType = getMostGeneralBaseType(xShapedType, yShapedType);
			SmallVector<mlir::Value, 3> dynamicSizes;

			for (long j = 0; j < xShapedType.getRank(); j++)
				if (xShapedType.isDynamicDim(j))
					dynamicSizes.push_back(rewriter.create<mlir::DimOp>(location, x, j));

			result = rewriter.create<AllocaOp>(
					location,
					MemRefType::get(xShapedType.getShape(), resultBaseType));

			if (xShapedType.hasStaticShape() && yShapedType.hasStaticShape())
			{
				// If both the operands have a static shape, we can optimize the sum
				// by lowering the memrefs to vectors and apply SIMD operations.

				x = memRefToVector(&rewriter, x);
				y = memRefToVector(&rewriter, y);

				SmallVector<mlir::Value, 2> currentOperands = { x, y };
				auto castOp = rewriter.create<CastCommonOp>(location, currentOperands);

				x = castOp.getResult(0);
				y = castOp.getResult(1);

				mlir::Type type = getBaseType(castOp.type());

				mlir::Value zeroValue = rewriter.create<ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
				SmallVector<mlir::Value, 3> indexes(result.getType().cast<ShapedType>().getRank(), zeroValue);

				if (type.isIndex() || type.isa<IntegerType>())
					rewriter.create<AffineVectorStoreOp>(location, rewriter.create<AddIOp>(location, x, y), result, indexes);
				else if (type.isa<FloatType>())
					rewriter.create<AffineVectorStoreOp>(location, rewriter.create<AddFOp>(location, x, y), result, indexes);
				else
					return rewriter.notifyMatchFailure(op, "Incompatible types");
			}
			else
			{
				// Otherwise, we need to fallback to the generic case and operate
				// on each element at a time.

				mlir::Value zero = rewriter.create<ConstantOp>(location, rewriter.getZeroAttr(rewriter.getIndexType()));
				SmallVector<mlir::Value, 3> lowerBounds(xShapedType.getRank(), zero);
				SmallVector<mlir::Value, 3> upperBounds;
				SmallVector<long, 3> steps;

				for (long dimension = 0; dimension < xShapedType.getRank(); dimension++)
				{
					upperBounds.push_back(rewriter.create<mlir::DimOp>(location, x, dimension));
					steps.push_back(1);
				}

				buildAffineLoopNest(
						rewriter, location, lowerBounds, upperBounds, steps,
						[&](OpBuilder& nestedBuilder, Location loc, ValueRange ivs) {
							// Cast the operands to the most generic type
							x = rewriter.create<AffineLoadOp>(location, x, ivs);
							y = rewriter.create<AffineLoadOp>(location, y, ivs);

							auto castOp = rewriter.create<CastCommonOp>(location, ValueRange({ x, y }));

							x = castOp.getResult(0);
							y = castOp.getResult(1);

							// Sum the values and store the result
							mlir::Type sumType = castOp.type();

							if (sumType.isIndex() || sumType.isa<IntegerType>())
								rewriter.create<AffineStoreOp>(location, rewriter.create<AddIOp>(location, x, y), result, ivs);
							else if (sumType.isa<FloatType>())
								rewriter.create<AffineStoreOp>(location, rewriter.create<AddFOp>(location, x, y), result, ivs);
						});
			}
		}
		else
		{
			SmallVector<mlir::Value, 2> currentOperands = { x, y };
			auto castOp = rewriter.create<CastCommonOp>(location, currentOperands);

			x = castOp.getResult(0);
			y = castOp.getResult(1);

			mlir::Type type = castOp.type();

			while (type.isa<ShapedType>())
				type = type.cast<ShapedType>().getElementType();

			if (type.isIndex() || type.isa<IntegerType>())
				result = rewriter.create<AddIOp>(location, x, y);
			else if (type.isa<FloatType>())
				result = rewriter.create<AddFOp>(location, x, y);
			else
				return rewriter.notifyMatchFailure(op, "Incompatible types");
		}
	}

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult SubOpLowering::matchAndRewrite(SubOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		SmallVector<mlir::Value, 2> currentOperands = { result, operands[i] };
		auto castOp = rewriter.create<CastCommonOp>(location, currentOperands);
		auto castedOperands = castOp.getResults();
		mlir::Type type = castOp.type();

		while (type.isa<ShapedType>())
			type = type.cast<ShapedType>().getElementType();

		if (type.isIndex() || type.isa<IntegerType>())
			result = rewriter.create<SubIOp>(location, castedOperands[0], castedOperands[1]);
		else if (type.isa<FloatType>())
			result = rewriter.create<SubFOp>(location, castedOperands[0], castedOperands[1]);
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");
	}

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

/**
 * Lower the multiplication operation to the vector or standard dialect.
 * According to the elements types and shapes, there are different scenarios:
 *
 * 1. scalar * scalar = scalar
 *    Normal multiplication, it is lowered to the standard dialect by just
 *    looking at their types
 * 2. scalar * vector[n] = vector[n]
 *    Scalar multiplication. It can be seen as a special case of the outer
 *    product of the vector dialect.
 * 3. vector[n] * vector[n] = scalar
 *    Cross product. It can be seen as a degenerate case of matrix product
 *    with the common dimension set to 1.
 * 4. vector[n] * matrix[n,m] = vector[m]
 *    The first vector is multiplied for each column of the matrix, thus
 *    generating each column of the result vector. It can be seen as
 *    transpose(vector[n]) * matrix[n,m] = transpose(vector[m])
 * 5. matrix[n,m] * vector[m] = vector[n]
 *    Cross product.
 * 6. matrix[n,m] * matrix[m,p] = matrix[n,p]
 *    Cross product.
 */
LogicalResult MulOpLowering::matchAndRewrite(MulOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		SmallVector<mlir::Value, 2> currentOperands = { result, operands[i] };
		auto castOp = rewriter.create<CastCommonOp>(location, currentOperands);

		mlir::Value x = castOp.getResults()[0];
		mlir::Value y = castOp.getResults()[1];

		mlir::Type xType = x.getType();
		mlir::Type yType = y.getType();

		if (!xType.isa<ShapedType>() && !yType.isa<ShapedType>())
		{
			// Case 1: scalar * scalar = scalar
			assert(x.getType() == y.getType());

			if (x.getType().isIndex() || x.getType().isa<IntegerType>())
				result = rewriter.create<MulIOp>(location, x, y);
			else if (x.getType().isa<FloatType>())
				result = rewriter.create<MulFOp>(location, x, y);
			else
				return rewriter.notifyMatchFailure(op, "Incompatible types");
		}
		else if (!xType.isa<ShapedType>() && yType.isa<ShapedType>())
		{
			// Case 2: scalar * vector[n] = vector[n]
			mlir::Value zeroValue = rewriter.create<ConstantOp>(location, rewriter.getZeroAttr(yType));
			result = rewriter.create<mlir::vector::OuterProductOp>(location, y, x, zeroValue);
		}
		else if (xType.isa<ShapedType>() && !yType.isa<ShapedType>())
		{
			// Case 2: vector[n] * scalar = vector[n]
			mlir::Value zeroValue = rewriter.create<ConstantOp>(location, rewriter.getZeroAttr(xType));
			result = rewriter.create<mlir::vector::OuterProductOp>(location, x, y, zeroValue);
		}
		else if (xType.isa<ShapedType>() && yType.isa<ShapedType>())
		{
			result = rewriter.create<CrossProductOp>(location, x, y);
		}
		else
		{
			return rewriter.notifyMatchFailure(op, "Incompatible types");
		}
	}

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

/**
 * Lower the cross product operation to the vector or SCF dialect.
 * According to the elements shapes, there are different scenarios:
 *
 * 1. vector[n] * vector[n] = scalar
 *    Cross product. It can be seen as a degenerate case of matrix product
 *    with the common dimension set to 1.
 * 2. vector[n] * matrix[n,m] = vector[m]
 *    The first vector is multiplied for each column of the matrix, thus
 *    generating each column of the result vector. It can be seen as
 *    transpose(vector[n]) * matrix[n,m] = transpose(vector[m])
 * 3. matrix[n,m] * vector[m] = vector[n]
 *    Cross product.
 * 4. matrix[n,m] * matrix[m,p] = matrix[n,p]
 *    Cross product.
 */
LogicalResult CrossProductOpLowering::matchAndRewrite(CrossProductOp op, PatternRewriter& rewriter) const
{
	auto location = op.getLoc();

	mlir::Value x = op.lhs();
	mlir::Value y = op.rhs();

	auto xType = x.getType().cast<ShapedType>();
	auto yType = y.getType().cast<ShapedType>();

	if (xType.getRank() == 1 && yType.getRank() == 1)
	{
		//rewriter.create<scf::ParallelOp>(location)
	}

	/*
	if (xShapedType.getRank() == 1 && yShapedType.getRank() == 1 &&
			xShapedType.getDimSize(0) == yShapedType.getDimSize(0))
	{
		// Case 3: vector[n] * vector[n] = scalar
		// TODO
	}
	 */
}

LogicalResult DivOpLowering::matchAndRewrite(DivOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value result = operands[0];

	for (size_t i = 1; i < operands.size(); i++)
	{
		SmallVector<mlir::Value, 2> currentOperands = { result, operands[i] };
		auto castOp = rewriter.create<CastCommonOp>(location, currentOperands);
		auto castedOperands = castOp.getResults();
		mlir::Type type = castOp.type();

		while (type.isa<ShapedType>())
			type = type.cast<ShapedType>().getElementType();

		if (type.isIndex() || type.isa<IntegerType>())
			result = rewriter.create<SignedDivIOp>(location, castedOperands[0], castedOperands[1]);
		else if (type.isa<FloatType>())
			result = rewriter.create<DivFOp>(location, castedOperands[0], castedOperands[1]);
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");
	}

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}
