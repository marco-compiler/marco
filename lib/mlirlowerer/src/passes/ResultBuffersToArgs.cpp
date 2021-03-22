#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>
#include <modelica/mlirlowerer/passes/ResultBuffersToArgs.h>

using namespace modelica;

static void updateFuncOp(mlir::FuncOp func, llvm::SmallVectorImpl<mlir::BlockArgument>& appendedEntryArgs)
{
	auto functionType = func.getType();

	// Collect information about the results will become appended arguments.
	llvm::SmallVector<mlir::Type, 6> erasedResultTypes;
	llvm::SmallVector<unsigned int, 6> erasedResultIndices;

	for (auto resultType : llvm::enumerate(functionType.getResults())) {
		if (auto pointerType = resultType.value().cast<PointerType>(); pointerType && pointerType.hasConstantShape()) {
			erasedResultIndices.push_back(resultType.index());
			erasedResultTypes.push_back(PointerType::get(pointerType.getContext(), BufferAllocationScope::unknown, pointerType.getElementType(), pointerType.getShape()));
		}
	}

	// Add the new arguments to the function type.
	auto newArgTypes = llvm::to_vector<6>(
			llvm::concat<const mlir::Type>(functionType.getInputs(), erasedResultTypes));
	auto newFunctionType = mlir::FunctionType::get(func.getContext(), newArgTypes, functionType.getResults());
	func.setType(newFunctionType);

	// Transfer the result attributes to arg attributes.
	for (int i = 0, e = erasedResultTypes.size(); i < e; i++)
		func.setArgAttrs(functionType.getNumInputs() + i,
										 func.getResultAttrs(erasedResultIndices[i]));

	// Erase the results.
	func.eraseResults(erasedResultIndices);

	// Add the new arguments to the entry block if the function is not external.
	if (func.isExternal())
		return;

	auto newArgs = func.front().addArguments(erasedResultTypes);
	appendedEntryArgs.append(newArgs.begin(), newArgs.end());
}

static void updateReturnOps(mlir::FuncOp func, llvm::ArrayRef<mlir::BlockArgument> appendedEntryArgs) {
	func.walk([&](mlir::ReturnOp op) {
		llvm::SmallVector<mlir::Value, 6> copyIntoOutParams;
		llvm::SmallVector<mlir::Value, 6> keepAsReturnOperands;


		for (mlir::Value operand : op.getOperands())
		{
			if (auto pointerType = operand.getType().cast<PointerType>(); pointerType && pointerType.hasConstantShape())
			{
				copyIntoOutParams.push_back(operand);

				/*
				auto allocaOp = operand.getDefiningOp<LoadOp>().memory().getDefiningOp<AllocaOp>();

				builder.setInsertionPointAfter(allocaOp);
				auto newOp = builder.create<AllocaOp>(op.getLoc(),
																							PointerType::get(op->getContext(), BufferAllocationScope::stack, pointerType.getElementType(), pointerType.getShape()));
				allocaOp->replaceAllUsesWith(newOp);
				 */

				/*
				llvm::SmallVector<StoreOp, 3> storeOps;

				for (auto user : allocaOp->getUsers())
				{
					llvm::errs() << "AAA ";
					user->dump();
					user->erase();

					user->walk([&](mlir::StoreOp storeOp) {
						llvm::errs() << "BBB ";
						storeOp->dump();
					});
				}
				allocaOp->erase();
				 */
			}
			else
			{
				keepAsReturnOperands.push_back(operand);
			}
		}

		mlir::OpBuilder builder(op);

		for (auto t : llvm::zip(copyIntoOutParams, appendedEntryArgs))
		{
			auto source = std::get<0>(t);
			auto dest = std::get<0>(t);
			auto pointerType = dest.getType().cast<PointerType>();

			mlir::Value zero = builder.create<mlir::ConstantOp>(op.getLoc(), builder.getIndexAttr(0));
			mlir::Value one = builder.create<mlir::ConstantOp>(op.getLoc(), builder.getIndexAttr(1));

			llvm::SmallVector<mlir::Value, 3> lowerBounds(pointerType.getRank(), zero);
			llvm::SmallVector<mlir::Value, 3> upperBounds;
			llvm::SmallVector<mlir::Value, 3> steps(pointerType.getRank(), one);

			for (unsigned int i = 0, e = pointerType.getRank(); i < e; ++i)
			{
				mlir::Value dim = builder.create<mlir::ConstantOp>(op->getLoc(), builder.getIndexAttr(i));
				upperBounds.push_back(builder.create<DimOp>(op.getLoc(), dest, dim));
			}

			// Create nested loops in order to iterate on each dimension of the array
			mlir::scf::buildLoopNest(
					builder, op.getLoc(), lowerBounds, upperBounds, steps, llvm::None,
					[&](mlir::OpBuilder& nestedBuilder, mlir::Location loc, mlir::ValueRange position, mlir::ValueRange args) -> std::vector<mlir::Value> {
						mlir::Value val = builder.create<LoadOp>(op->getLoc(), source, position);
						builder.create<StoreOp>(op.getLoc(), val, dest, position);
						return std::vector<mlir::Value>();
					});
		}


		/*
			builder.create<linalg::CopyOp>(op.getLoc(), std::get<0>(t),std::get<1>(t));
		 */

		builder.setInsertionPointAfter(op);
		builder.create<mlir::ReturnOp>(op.getLoc(), keepAsReturnOperands);
		op.erase();
	});
}

static mlir::LogicalResult updateCalls(mlir::ModuleOp module) {
	bool didFail = false;

	module.walk([&](CallOp op) {
		llvm::SmallVector<mlir::Value, 6> replaceWithNewCallResults;
		llvm::SmallVector<mlir::Value, 6> replaceWithOutParams;

		for (mlir::OpResult result : op.getResults()) {
			if (auto pointerType = result.getType().cast<PointerType>(); pointerType && pointerType.hasConstantShape())
				replaceWithOutParams.push_back(result);
			else
				replaceWithNewCallResults.push_back(result);
		}

		llvm::SmallVector<mlir::Value, 6> outParams;
		mlir::OpBuilder builder(op);

		for (mlir::Value ptr : replaceWithOutParams) {
			auto pointerType = ptr.getType().cast<PointerType>();

			if (!pointerType.hasConstantShape()) {
				op.emitError() << "cannot create out param for dynamically shaped buffer";
				didFail = true;
				return;
			}

			mlir::Value outParam = builder.create<AllocaOp>(op.getLoc(), pointerType.getElementType(), pointerType.getShape());
			ptr.replaceAllUsesWith(outParam);
			outParams.push_back(outParam);
		}

		auto newOperands = llvm::to_vector<6>(op.getOperands());
		newOperands.append(outParams.begin(), outParams.end());

		auto newResultTypes = llvm::to_vector<6>(llvm::map_range(
				replaceWithNewCallResults, [](mlir::Value v) { return v.getType(); }));

		auto newCall = builder.create<CallOp>(op.getLoc(), op.callee(), newResultTypes, newOperands, 0);

		for (auto t : llvm::zip(replaceWithNewCallResults, newCall.getResults()))
			std::get<0>(t).replaceAllUsesWith(std::get<1>(t));

		op.erase();
	});

	return mlir::failure(didFail);
}

class ResultBuffersToArgsPass: public mlir::PassWrapper<ResultBuffersToArgsPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void getDependentDialects(mlir::DialectRegistry &registry) const override
	{
		registry.insert<mlir::scf::SCFDialect>();
	}

	void runOnOperation() override
	{
		auto module = getOperation();

		for (auto func : module.getOps<mlir::FuncOp>()) {
			llvm::SmallVector<mlir::BlockArgument, 6> appendedEntryArgs;
			updateFuncOp(func, appendedEntryArgs);

			if (func.isExternal())
				continue;

			updateReturnOps(func, appendedEntryArgs);
		}

		if (failed(updateCalls(module)))
			return signalPassFailure();
	}
};

std::unique_ptr<mlir::Pass> modelica::createResultBuffersToArgsPass()
{
	return std::make_unique<ResultBuffersToArgsPass>();
}
