#include <mlir/Conversion/Passes.h>
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
		if (auto pointerType = resultType.value().dyn_cast<PointerType>(); pointerType && pointerType.hasConstantShape()) {
			erasedResultIndices.push_back(resultType.index());
			erasedResultTypes.push_back(PointerType::get(pointerType.getContext(), BufferAllocationScope::unknown, pointerType.getElementType(), pointerType.getShape()));
		}
	}

	// Add the new arguments to the function type
	auto newArgTypes = llvm::to_vector<6>(
			llvm::concat<const mlir::Type>(functionType.getInputs(), erasedResultTypes));
	auto newFunctionType = mlir::FunctionType::get(func.getContext(), newArgTypes, functionType.getResults());
	func.setType(newFunctionType);

	// Transfer the result attributes to arg attributes
	for (int i = 0, e = erasedResultTypes.size(); i < e; i++)
		func.setArgAttrs(functionType.getNumInputs() + i,
										 func.getResultAttrs(erasedResultIndices[i]));

	// Erase the results
	func.eraseResults(erasedResultIndices);

	// Add the new arguments to the entry block if the function is not external
	if (func.isExternal())
		return;

	auto newArgs = func.front().addArguments(erasedResultTypes);
	appendedEntryArgs.append(newArgs.begin(), newArgs.end());
}

static void updateReturnOps(mlir::FuncOp func, llvm::ArrayRef<mlir::BlockArgument> appendedEntryArgs) {
	func.walk([&](mlir::ReturnOp op) {
		size_t entryArgCounter = 0;
		llvm::SmallVector<mlir::Value, 6> returnOperands;

		for (mlir::Value operand : op.getOperands())
		{
			if (auto pointerType = operand.getType().dyn_cast<PointerType>(); pointerType && pointerType.hasConstantShape())
			{
				auto allocaOp = operand.getDefiningOp<LoadOp>().memory().getDefiningOp<AllocaOp>();

				for (auto* user : allocaOp->getUsers())
				{
					if (mlir::isa<LoadOp>(user))
					{
						user->replaceAllUsesWith(mlir::ValueRange(appendedEntryArgs[entryArgCounter]));
						user->erase();
					}
					else if (mlir::isa<StoreOp>(user))
					{
						auto allocOp = mlir::cast<StoreOp>(user).value().getDefiningOp<AllocOp>();
						allocOp->remove();
						user->remove();
					}
				}

				allocaOp->remove();
				entryArgCounter++;
			}
			else
			{
				returnOperands.push_back(operand);
			}
		}

		mlir::OpBuilder builder(op);
		builder.create<mlir::ReturnOp>(op.getLoc(), returnOperands);
		op.erase();
	});
}

static void updateSubscriptionOps(mlir::FuncOp func) {
	func.walk([&](SubscriptionOp op) {
		if (op.resultType().getAllocationScope() != op.source().getType().cast<PointerType>().getAllocationScope())
		{
			mlir::OpBuilder builder(op);
			mlir::Value newOp = builder.create<SubscriptionOp>(op->getLoc(), op.source(), op.indexes());
			op.replaceAllUsesWith(newOp);
			op->erase();
		}
	});
}

static mlir::LogicalResult updateCalls(mlir::ModuleOp module) {
	bool didFail = false;

	module.walk([&](CallOp op) {
		llvm::SmallVector<mlir::Value, 6> replaceWithNewCallResults;
		llvm::SmallVector<mlir::Value, 6> replaceWithOutParams;

		for (mlir::OpResult result : op.getResults()) {
			if (auto pointerType = result.getType().dyn_cast<PointerType>(); pointerType && pointerType.hasConstantShape())
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

			outParam = builder.create<PtrCastOp>(
					op->getLoc(), outParam,
					PointerType::get(pointerType.getContext(), BufferAllocationScope::unknown, pointerType.getElementType(), pointerType.getShape()));

			outParams.push_back(outParam);
		}

		auto newOperands = llvm::to_vector<6>(op.getOperands());
		newOperands.append(outParams.begin(), outParams.end());

		auto newResultTypes = llvm::to_vector<6>(llvm::map_range(
				replaceWithNewCallResults, [](mlir::Value v) { return v.getType(); }));

		auto newCall = builder.create<CallOp>(op.getLoc(), op.callee(), newResultTypes, newOperands, replaceWithOutParams.size());

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
			updateSubscriptionOps(func);
		}

		if (failed(updateCalls(module)))
			return signalPassFailure();
	}
};

std::unique_ptr<mlir::Pass> modelica::createResultBuffersToArgsPass()
{
	return std::make_unique<ResultBuffersToArgsPass>();
}
