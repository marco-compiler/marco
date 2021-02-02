#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.hpp>
#include <modelica/mlirlowerer/LowerToStandard.hpp>

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

LogicalResult NegateOpLowering::matchAndRewrite(NegateOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	mlir::Value operand = op->getOperand(0);

	mlir::Type type = op->getResultTypes()[0];

	if (type.isa<MemRefType>())
	{
		auto memRefType = type.cast<MemRefType>();

		mlir::Value fake = rewriter.create<AllocaOp>(location, MemRefType::get({ 2 }, rewriter.getI1Type()));
		rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true)).getResult(), fake, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0)).getResult());
		rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false)).getResult(), fake, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(1)).getResult());
		operand = fake;

		mlir::VectorType vectorType = VectorType::get(memRefType.getShape(), memRefType.getElementType());
		mlir::Value zeroValue = rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0));
		SmallVector<mlir::Value, 3> indexes(memRefType.getRank(), zeroValue);
		mlir::Value vector = rewriter.create<AffineVectorLoadOp>(location, vectorType, operand, indexes);
		rewriter.create<mlir::vector::PrintOp>(location, vector);

		SmallVector<bool, 3> trueValues(memRefType.getNumElements(), true);
		mlir::Value trueVector = rewriter.create<ConstantOp>(location, rewriter.getBoolVectorAttr(trueValues));
		mlir::Value xorOp = rewriter.create<XOrOp>(location, vector, trueVector);
		rewriter.create<mlir::vector::PrintOp>(location, xorOp);

		mlir::Value destination = rewriter.create<AllocaOp>(location, memRefType);
		rewriter.create<AffineVectorStoreOp>(location, xorOp, destination, indexes);

		//mlir::Value unranked = rewriter.create<MemRefCastOp>(location, destination, MemRefType::get(-1, rewriter.getI32Type()));
		//rewriter.create<CallOp>(location, "print_memref_i32", TypeRange(), unranked);

		/*
		rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false)).getResult(), destination, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(0)).getResult());
		rewriter.create<StoreOp>(location, rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true)).getResult(), destination, rewriter.create<ConstantOp>(location, rewriter.getIndexAttr(1)).getResult());
		mlir::Value vectorAfterStore = rewriter.create<AffineVectorLoadOp>(location, vectorType, destination, indexes);
		rewriter.create<mlir::vector::PrintOp>(location, vectorAfterStore);
		 */

		rewriter.replaceOp(op, destination);
	}
	else
	{
		mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
		rewriter.replaceOpWithNewOp<XOrOp>(op, operand, trueValue);
	}

	//op->getParentOp()->dump();

	return success();
}

LogicalResult AddOpLowering::matchAndRewrite(AddOp op, PatternRewriter& rewriter) const
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
			result = rewriter.create<AddIOp>(location, castedOperands[0], castedOperands[1]);
		else if (type.isa<FloatType>())
			result = rewriter.create<AddFOp>(location, castedOperands[0], castedOperands[1]);
		else
			return rewriter.notifyMatchFailure(op, "Incompatible types");
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
	mlir::Value x = op.lhs();
	mlir::Value y = op.rhs();

	auto xType = x.getType().cast<ShapedType>();
	auto yType = y.getType().cast<ShapedType>();

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

LogicalResult EqOpLowering::matchAndRewrite(EqOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::eq, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OEQ, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OEQ, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OEQ, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult NotEqOpLowering::matchAndRewrite(NotEqOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::ne, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::ONE, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::ONE, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::ONE, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult GtOpLowering::matchAndRewrite(GtOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::sgt, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGT, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGT, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGT, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult GteOpLowering::matchAndRewrite(GteOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::sge, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGE, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGE, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OGE, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult LtOpLowering::matchAndRewrite(LtOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::slt, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLT, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLT, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLT, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult LteOpLowering::matchAndRewrite(LteOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto operands = op->getOperands();

	mlir::Value lhs = operands[0];
	mlir::Type lhsBaseType = lhs.getType();

	while (lhsBaseType.isa<ShapedType>())
		lhsBaseType = lhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value rhs = operands[1];
	mlir::Type rhsBaseType = rhs.getType();

	while (rhsBaseType.isa<ShapedType>())
		rhsBaseType = rhsBaseType.cast<ShapedType>().getElementType();

	mlir::Value result;

	if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<IntegerType>())
		result = rewriter.create<CmpIOp>(location, CmpIPredicate::sle, lhs, rhs);
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<FloatType>())
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLE, lhs, rhs);
	else if (lhsBaseType.isa<IntegerType>() && rhsBaseType.isa<FloatType>())
	{
		lhs = rewriter.create<CastOp>(location, lhs, rhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLE, lhs, rhs);
	}
	else if (lhsBaseType.isa<FloatType>() && rhsBaseType.isa<IntegerType>())
	{
		rhs = rewriter.create<CastOp>(location, rhs, lhs.getType());
		result = rewriter.create<CmpFOp>(location, CmpFPredicate::OLE, lhs, rhs);
	}
	else
		return rewriter.notifyMatchFailure(op, "Incompatible types");

	rewriter.replaceOpWithNewOp<CastOp>(op, result, op->getResultTypes()[0]);
	return success();
}

LogicalResult IfOpLowering::matchAndRewrite(IfOp op, PatternRewriter& rewriter) const
{
	bool hasElseBlock = !op.elseRegion().empty();

	auto thenBuilder = [&](mlir::OpBuilder& builder, mlir::Location location)
	{
		rewriter.mergeBlocks(&op.thenRegion().front(), rewriter.getInsertionBlock());
	};

	if (hasElseBlock)
	{
		rewriter.create<scf::IfOp>(
				op.getLoc(), TypeRange(), op.condition(), thenBuilder,
				[&](mlir::OpBuilder& builder, mlir::Location location)
				{
					rewriter.mergeBlocks(&op.elseRegion().front(), builder.getInsertionBlock());
				});
	}
	else
	{
		rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(), op.condition(), thenBuilder, nullptr);
	}

	rewriter.eraseOp(op);
	return success();
}

LogicalResult ForOpLowering::matchAndRewrite(ForOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();

	// Split the current block
	Block* currentBlock = rewriter.getInsertionBlock();
	Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

	// Inline regions
	Block* conditionBlock = &op.condition().front();
	Block* bodyBlock = &op.body().front();
	Block* stepBlock = &op.step().front();

	rewriter.inlineRegionBefore(op.step(), continuation);
	rewriter.inlineRegionBefore(op.body(), stepBlock);
	rewriter.inlineRegionBefore(op.condition(), bodyBlock);

	// Start the for loop by branching to the "condition" region
	rewriter.setInsertionPointToEnd(currentBlock);
	rewriter.create<BranchOp>(location, conditionBlock, op.args());

	// The loop is supposed to be breakable. Thus, before checking the normal
	// condition, we first need to check if the break condition variable has
	// been set to true in the previous loop execution. If it is set to true,
	// it means that a break statement has been executed and thus the loop
	// must be terminated.

	rewriter.setInsertionPointToStart(conditionBlock);

	mlir::Value breakCondition = rewriter.create<LoadOp>(location, op.breakCondition());
	mlir::Value returnCondition = rewriter.create<LoadOp>(location, op.returnCondition());
	mlir::Value stopCondition = rewriter.create<OrOp>(location, breakCondition, returnCondition);

	mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
	mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

	auto ifOp = rewriter.create<scf::IfOp>(location, rewriter.getI1Type(), condition, true);
	Block* originalCondition = rewriter.splitBlock(conditionBlock, rewriter.getInsertionPoint());

	// If the break condition variable is set to true, return false from the
	// condition block in order to stop the loop execution.
	rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
	mlir::Value falseValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false));
	rewriter.create<scf::YieldOp>(location, falseValue);

	// Move the original condition check in the "else" branch
	rewriter.mergeBlocks(originalCondition, &ifOp.elseRegion().front());
	rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
	auto conditionOp = cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
	rewriter.replaceOpWithNewOp<scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

	// The original condition operation is converted to the SCF one and takes
	// as condition argument the result of the If operation, which is false
	// if a break must be executed or the intended condition value otherwise.
	rewriter.setInsertionPointAfter(ifOp);
	rewriter.create<CondBranchOp>(location, ifOp.getResult(0),
																bodyBlock, conditionOp.args(), continuation, ValueRange());

	// Replace "body" block terminator with a branch to the "step" block
	rewriter.setInsertionPointToEnd(bodyBlock);
	auto bodyYieldOp = cast<YieldOp>(bodyBlock->getTerminator());
	rewriter.replaceOpWithNewOp<BranchOp>(bodyYieldOp, stepBlock, bodyYieldOp->getOperands());

	// Branch to the condition check after incrementing the induction variable
	rewriter.setInsertionPointToEnd(stepBlock);
	auto stepYieldOp = cast<YieldOp>(stepBlock->getTerminator());
	rewriter.replaceOpWithNewOp<BranchOp>(stepYieldOp, conditionBlock, stepYieldOp->getOperands());

	rewriter.eraseOp(op);
	return success();
}

LogicalResult WhileOpLowering::matchAndRewrite(WhileOp op, PatternRewriter& rewriter) const
{
	Location location = op.getLoc();
	auto whileOp = rewriter.create<scf::WhileOp>(location, TypeRange(), ValueRange());

	// The body block requires no modification apart from the change of the
	// terminator to the SCF dialect one.

	rewriter.createBlock(&whileOp.after());
	rewriter.mergeBlocks(&op.body().front(), &whileOp.after().front());
	mlir::Block* body = &whileOp.after().front();
	auto bodyTerminator = cast<YieldOp>(body->getTerminator());
	rewriter.setInsertionPointToEnd(body);
	rewriter.replaceOpWithNewOp<scf::YieldOp>(bodyTerminator, bodyTerminator.getOperands());

	// The loop is supposed to be breakable. Thus, before checking the normal
	// condition, we first need to check if the break condition variable has
	// been set to true in the previous loop execution. If it is set to true,
	// it means that a break statement has been executed and thus the loop
	// must be terminated.

	rewriter.createBlock(&whileOp.before());
	rewriter.setInsertionPointToStart(&whileOp.before().front());

	mlir::Value breakCondition = rewriter.create<LoadOp>(location, op.breakCondition());
	mlir::Value returnCondition = rewriter.create<LoadOp>(location, op.returnCondition());
	mlir::Value stopCondition = rewriter.create<OrOp>(location, breakCondition, returnCondition);

	mlir::Value trueValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(true));
	mlir::Value condition = rewriter.create<EqOp>(location, stopCondition, trueValue);

	auto ifOp = rewriter.create<scf::IfOp>(location, rewriter.getI1Type(), condition, true);

	// If the break condition variable is set to true, return false from the
	// condition block in order to stop the loop execution.
	rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
	mlir::Value falseValue = rewriter.create<ConstantOp>(location, rewriter.getBoolAttr(false));
	rewriter.create<scf::YieldOp>(location, falseValue);

	// Move the original condition check in the "else" branch
	rewriter.mergeBlocks(&op.condition().front(), &ifOp.elseRegion().front());
	rewriter.setInsertionPointToEnd(&ifOp.elseRegion().front());
	auto conditionOp = cast<ConditionOp>(ifOp.elseRegion().front().getTerminator());
	rewriter.replaceOpWithNewOp<scf::YieldOp>(conditionOp, conditionOp.getOperand(0));

	// The original condition operation is converted to the SCF one and takes
	// as condition argument the result of the If operation, which is false
	// if a break must be executed or the intended condition value otherwise.
	rewriter.setInsertionPointAfter(ifOp);
	rewriter.create<scf::ConditionOp>(location, ifOp.getResult(0), conditionOp.args());

	rewriter.eraseOp(op);
	return success();
}

LogicalResult YieldOpLowering::matchAndRewrite(YieldOp op, PatternRewriter& rewriter) const
{
	rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
	return success();
}

void ModelicaToStandardLoweringPass::runOnOperation()
{
	auto module = getOperation();
	ConversionTarget target(getContext());

	target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp>();
	target.addLegalDialect<StandardOpsDialect>();
	target.addLegalDialect<linalg::LinalgDialect>();
	target.addLegalDialect<mlir::vector::VectorDialect>();

	// The Modelica dialect is defined as illegal, so that the conversion
	// will fail if any of its operations are not converted.
	target.addIllegalDialect<ModelicaDialect>();

	// Provide the set of patterns that will lower the Modelica operations
	mlir::OwningRewritePatternList patterns;
	populateModelicaToStandardConversionPatterns(patterns, &getContext());
	populateLoopToStdConversionPatterns(patterns, &getContext());
	populateAffineToVectorConversionPatterns(patterns, &getContext());

	// With the target and rewrite patterns defined, we can now attempt the
	// conversion. The conversion will signal failure if any of our "illegal"
	// operations were not converted successfully.
	if (failed(applyFullConversion(module, target, move(patterns))))
		signalPassFailure();
}

void modelica::populateModelicaToStandardConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* context)
{
	// Generic operations
	patterns.insert<CastOpLowering>(context);
	patterns.insert<CastCommonOpLowering>(context);
	patterns.insert<AssignmentOpLowering>(context);

	// Math operations
	patterns.insert<NegateOpLowering>(context);
	patterns.insert<AddOpLowering>(context);
	patterns.insert<SubOpLowering>(context);
	patterns.insert<MulOpLowering>(context);
	patterns.insert<CrossProductOpLowering>(context);
	patterns.insert<DivOpLowering>(context);

	// Comparison operations
	patterns.insert<EqOpLowering>(context);
	patterns.insert<NotEqOpLowering>(context);
	patterns.insert<GtOpLowering>(context);
	patterns.insert<GteOpLowering>(context);
	patterns.insert<LtOpLowering>(context);
	patterns.insert<LteOpLowering>(context);

	// Control flow operations
	patterns.insert<IfOpLowering>(context);
	patterns.insert<ForOpLowering>(context);
	patterns.insert<WhileOpLowering>(context);
	patterns.insert<YieldOpLowering>(context);
}

std::unique_ptr<mlir::Pass> modelica::createModelicaToStandardLoweringPass()
{
	return std::make_unique<ModelicaToStandardLoweringPass>();
}
