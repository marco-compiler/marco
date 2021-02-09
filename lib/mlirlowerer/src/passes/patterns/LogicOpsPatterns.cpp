#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/patterns/LogicOpsPatterns.h>

using namespace mlir;
using namespace modelica;
using namespace std;

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
