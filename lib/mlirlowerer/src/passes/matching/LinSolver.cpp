#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/passes/matching/LinSolver.h>
#include <modelica/mlirlowerer/passes/model/Equation.h>
#include <modelica/mlirlowerer/passes/model/Expression.h>
#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/ReferenceMatcher.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/utils/IndexSet.hpp>
#include <numeric>

using namespace modelica::codegen;
using namespace model;

/**
 * Get the expression that represents the variables explicitated by source
 * and replace each occurence of that variable inside destination with the
 * aformentioned expression.
 *
 * @param builder			operation builder
 * @param source			equation containing the explicitated variable to be replaced
 * @param destination equation inside which the source variable occurences have to be replaced
 */
static void replaceUses(mlir::OpBuilder& builder, const Equation& source, Equation& destination)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	auto var = source.getDeterminedVariable();
	ReferenceMatcher matcher(destination);

	for (auto& access : matcher)
	{
		auto pathToVar = AccessToVar::fromExp(access.getExp());

		if (pathToVar.getVar() != var.getVar())
			continue;

		auto composedSource = source.composeAccess(pathToVar.getAccess());

		// Map the old induction values with the ones in the new equation
		mlir::BlockAndValueMapping mapper;

		for (auto [oldInduction, newInduction] :
				llvm::zip(composedSource.getOp().inductions(), destination.getOp().inductions()))
			mapper.map(oldInduction, newInduction);

		// Copy all the operations from the explicitated equation into the
		// one whose member has to be replaced.
		builder.setInsertionPointToStart(destination.getOp().body());
		EquationSidesOp clonedTerminator;

		for (auto& op : composedSource.getOp().body()->getOperations())
		{
			mlir::Operation* clonedOp = builder.clone(op, mapper);

			if (auto terminator = mlir::dyn_cast<EquationSidesOp>(clonedOp))
				clonedTerminator = terminator;
		}

		// Remove the cloned terminator. In fact, in the generic case we need
		// to preserve the original left-hand and right-hand sides of the
		// equations. If the member to be replaced is the same as a side of
		// the original equations, if will be automatically replaced inside
		// the remaining block terminator.
		clonedTerminator.erase();

		// Replace the uses of the value we want to replace.
		for (auto& use : destination.reachExp(access).getOp()->getUses())
		{
			// We need to check if we are inside the equation body block. In fact,
			// if the value to be replaced is an array (and not a scalar or a
			// subscription), we would replace the array instantiation itself,
			// which is outside the simulation block and thus would impact also
			// other equations.
			if (!destination.getOp()->isAncestor(use.getOwner()))
				continue;

			if (auto loadOp = mlir::dyn_cast<LoadOp>(use.getOwner()); loadOp.indexes().empty())
			{
				// If the value to be replaced is the declaration of a scalar
				// variable, we instead need to replace the load operations which
				// are executed on that variable.
				// Example:
				//  %0 = modelica.alloca : modelica.ptr<int>
				//  equation:
				//    %1 = modelica.load %0 : int
				//    modelica.equation_sides (%1, ...)
				// needs to become
				//  %0 = modelica.alloca : modelica.ptr<int>
				//  equation:
				//    modelica.equation_sides (%newValue, ...)

				loadOp->replaceAllUsesWith(clonedTerminator.rhs()[0].getDefiningOp());
				loadOp->erase();
			}
			else
			{
				use.set(clonedTerminator.rhs()[0]);
			}
		}

		composedSource.getOp()->erase();
	}

	destination.update();
}

static mlir::LogicalResult distributeMulAndDivOps(mlir::OpBuilder& builder, mlir::Operation* op)
{
	if (op == nullptr)
		return mlir::success();

	for (auto operand : op->getOperands())
		if (auto status = distributeMulAndDivOps(builder, operand.getDefiningOp()); failed(status))
			return status;

	if (auto distributableOp = mlir::dyn_cast<DistributableInterface>(op))
	{
		if (!mlir::isa<NegateOp>(op))
		{
			mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

			if (result != op)
				op->replaceAllUsesWith(result);
		}
	}

	return mlir::success();
}

static mlir::LogicalResult pushNegateOps(mlir::OpBuilder& builder, mlir::Operation* op)
{
	if (op == nullptr)
		return mlir::success();

	for (auto operand : op->getOperands())
		if (auto status = pushNegateOps(builder, operand.getDefiningOp()); failed(status))
			return status;

	if (auto distributableOp = mlir::dyn_cast<NegateOp>(op))
	{
		mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

		if (result != op)
			op->replaceAllUsesWith(result);
	}

	return mlir::success();
}

struct SubOpErasePattern : public mlir::OpRewritePattern<SubOp>
{
	using mlir::OpRewritePattern<SubOp>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(SubOp op, mlir::PatternRewriter& rewriter) const override
	{
		mlir::Value rhs = op.rhs();
		rhs = rewriter.create<NegateOp>(rhs.getLoc(), rhs.getType(), rhs);
		rewriter.replaceOpWithNewOp<AddOp>(op, op.resultType(), op.lhs(), rhs);
		return mlir::success();
	}
};

static mlir::LogicalResult removeSubtractions(EquationInterface equation)
{
	mlir::ConversionTarget target(*equation.getContext());
	target.addLegalDialect<ModelicaDialect>();
	target.addIllegalOp<SubOp>();

	mlir::OwningRewritePatternList patterns;
	patterns.insert<SubOpErasePattern>(equation->getContext());

	if (auto status = applyPartialConversion(equation, target, std::move(patterns)); failed(status))
		return status;

	return mlir::success();
}

static mlir::LogicalResult flattenSummedValues(mlir::Value value, llvm::SmallVectorImpl<mlir::Value>& values)
{
	if (auto addOp = mlir::dyn_cast<AddOp>(value.getDefiningOp()))
	{
		if (auto status = flattenSummedValues(addOp.lhs(), values); failed(status))
			return status;

		if (auto status = flattenSummedValues(addOp.rhs(), values); failed(status))
			return status;

		return mlir::success();
	}

	values.emplace_back(value);
	return mlir::success();
}

static bool usesMember(mlir::Value value, AccessToVar access)
{
	if (value == access.getVar())
		return true;

	mlir::Operation* op = value.getDefiningOp();

	if (mlir::isa<LoadOp, SubscriptionOp>(op))
	{
		auto subscriptionAccess = AccessToVar::fromExp(Expression::build(value));

		if (access == subscriptionAccess)
			return true;
	}

	if (auto negateOp = mlir::dyn_cast<NegateOp>(op))
		if (usesMember(negateOp.operand(), access))
			return true;

	if (auto mulOp = mlir::dyn_cast<MulOp>(op))
	{
		if (usesMember(mulOp.lhs(), access) ||
				usesMember(mulOp.lhs(), access))
			return true;
	}

	return false;
}

static mlir::Attribute getIntegerAttribute(mlir::OpBuilder& builder, mlir::Type type, int value)
{
	if (type.isa<BooleanType>())
		return BooleanAttribute::get(type, value);

	if (type.isa<IntegerType>())
		return IntegerAttribute::get(type, value);

	if (type.isa<RealType>())
		return RealAttribute::get(type, value);

	return builder.getIndexAttr(value);
}

static mlir::Value getMultiplyingFactor(mlir::OpBuilder& builder, mlir::Value value, AccessToVar access)
{
	if (value == access.getVar())
		return builder.create<ConstantOp>(value.getLoc(), getIntegerAttribute(builder, value.getType(), 1));

	mlir::Operation* op = value.getDefiningOp();

	if (mlir::isa<LoadOp, SubscriptionOp>(op))
	{
		auto subscriptionAccess = AccessToVar::fromExp(Expression::build(value));

		if (access == subscriptionAccess)
			return builder.create<ConstantOp>(op->getLoc(), getIntegerAttribute(builder, value.getType(), 1));
	}

	assert(mlir::isa<NegateOp>(op) || mlir::isa<MulOp>(op));

	if (auto negateOp = mlir::dyn_cast<NegateOp>(op))
		return builder.create<NegateOp>(
				negateOp.getLoc(), negateOp.resultType(),
				getMultiplyingFactor(builder, negateOp.operand(), access));

	if (auto mulOp = mlir::dyn_cast<MulOp>(op))
		return builder.create<MulOp>(
				mulOp.getLoc(), mulOp.resultType(),
				getMultiplyingFactor(builder, mulOp.lhs(), access),
				getMultiplyingFactor(builder, mulOp.rhs(), access));

	return value;
}

static mlir::Type getMostGenericType(mlir::Type x, mlir::Type y)
{
	if (x.isa<BooleanType>())
		return y;

	if (y.isa<BooleanType>())
		return x;

	if (x.isa<RealType>())
		return x;

	if (y.isa<RealType>())
		return y;

	if (x.isa<IntegerType>())
		return x;

	return y;
}

static mlir::LogicalResult groupLeftHand(mlir::OpBuilder& builder, Equation& equation)
{
	mlir::OpBuilder::InsertionGuard guard(builder);

	llvm::errs() << "BEFORE\n";
	equation.getOp()->dump();

	if (auto status = removeSubtractions(equation.getOp()); failed(status))
		return status;

	if (auto status = distributeMulAndDivOps(builder, equation.getTerminator().rhs()[0].getDefiningOp()); failed(status))
		return status;

	if (auto status = pushNegateOps(builder, equation.getTerminator().rhs()[0].getDefiningOp()); failed(status))
		return status;

	llvm::SmallVector<mlir::Value, 3> summedValues;

	if (auto status = flattenSummedValues(equation.getTerminator().rhs()[0], summedValues); failed(status))
		return status;

	auto access = AccessToVar::fromExp(equation.lhs());

	auto* pos = partition(summedValues, [&](auto value) {
		return usesMember(value, access);
	});

	builder.setInsertionPoint(equation.getTerminator());

	if (pos == summedValues.begin())
	{
		// There is nothing to be moved to the left-hand side of the equation
		return mlir::success();
	}

	if (pos == summedValues.end())
	{
		// All the right-hand side components should be moved to the left-hand
		// side and thus the variable will take value 0.
		auto terminator = equation.getTerminator();
		mlir::Value zeroValue = builder.create<ConstantOp>(equation.getOp().getLoc(), getIntegerAttribute(builder, terminator.lhs()[0].getType(), 0));
		builder.setInsertionPointAfter(terminator);
		builder.create<EquationSidesOp>(terminator.getLoc(), terminator.lhs(), zeroValue);
		terminator.erase();
		return mlir::success();
	}

	mlir::Value toBeMoved = std::accumulate(
			summedValues.begin(), pos,
			builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, summedValues[0].getType(), 0)).getResult(),
			[&](mlir::Value acc, mlir::Value value) -> mlir::Value {
				mlir::Value factor = getMultiplyingFactor(builder, value, access);
				return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, factor);
			});

	mlir::Value leftFactor = builder.create<SubOp>(
			toBeMoved.getLoc(), toBeMoved.getType(),
			builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, toBeMoved.getType(), 1)),
			toBeMoved);

	mlir::Value rhs = std::accumulate(
			summedValues.begin() + 1, summedValues.end(),
			builder.create<ConstantOp>(equation.getOp()->getLoc(), getIntegerAttribute(builder, summedValues[0].getType(), 0)).getResult(),
			[&](mlir::Value acc, mlir::Value value) -> mlir::Value {
				return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
			});

	rhs = builder.create<DivOp>(
			rhs.getLoc(),
			equation.getTerminator().lhs()[0].getType(),
			rhs, leftFactor);

	auto terminator = equation.getTerminator();
	builder.setInsertionPointAfter(terminator);
	builder.create<EquationSidesOp>(terminator.getLoc(), terminator.lhs(), rhs);
	terminator->erase();

	equation.update();

	llvm::errs() << "AFTER\n";
	equation.getOp()->dump();

	return mlir::success();
}

namespace modelica::codegen::model
{
	mlir::LogicalResult linearySolve(mlir::OpBuilder& builder, llvm::SmallVectorImpl<Equation>& equations)
	{
		for (auto eq = equations.rbegin(); eq != equations.rend(); ++eq)
			for (auto eq2 = eq + 1; eq2 != equations.rend(); ++eq2)
				replaceUses(builder, *eq, *eq2);

		for (auto& equation : equations)
			if (auto res = groupLeftHand(builder, equation); failed(res))
				return res;

		// Erase the useless operations that are remains of the previous
		// transformations.
		// TODO: avoid this pass and erase the orphan operations asap
		for (auto& equation : equations)
		{
			mlir::Block::reverse_iterator it(equation.getOp().body()->getTerminator());
			auto end = equation.getOp().body()->rend();

			while (it != end)
			{
				if (it->getNumResults() != 0 && it->getUses().empty())
				{
					// We can't just erase the operation, because we would invalidate
					// the iteration. Instead, we have to keep track of the current
					// operation, advance the iterator and only then erase the
					// operation.
					auto curr = it;
					++it;
					curr->erase();
				}
				else
				{
					++it;
				}
			}
		}

		return mlir::success();
	}
}
