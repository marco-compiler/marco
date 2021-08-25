#include <mlir/Transforms/BufferUtils.h>
#include <mlir/Transforms/Passes.h>
#include <marco/mlirlowerer/passes/BufferDeallocation.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaDialect.h>

using namespace mlir;
using namespace marco::codegen;
using namespace modelica;

namespace marco::codegen
{
	class BufferDeallocation : BufferPlacementTransformationBase
	{
		public:
		BufferDeallocation(Operation* op)
				: BufferPlacementTransformationBase(op),
					postDominators(op)
		{
		}

		void deallocate() const
		{
			for (const BufferPlacementAllocs::AllocEntry& entry : allocs)
			{
				Value alloc = std::get<0>(entry);
				auto aliasesSet = aliases.resolve(alloc);
				assert(!aliasesSet.empty() && "must contain at least one alias");

				// Determine the actual block to place the dealloc and get liveness
				// information.
				Block* placementBlock = findCommonDominator(alloc, aliasesSet, postDominators);
				const LivenessBlockInfo* livenessInfo = liveness.getLiveness(placementBlock);

				// We have to ensure that the dealloc will be after the last use of all
				// aliases of the given value. We first assume that there are no uses in
				// the placementBlock and that we can safely place the dealloc at the
				// beginning.
				Operation* endOperation = &placementBlock->front();

				// Iterate over all aliases and ensure that the endOperation will point
				// to the last operation of all potential aliases in the placementBlock.
				for (Value alias : aliasesSet)
				{
					// Ensure that the start operation is at least the defining operation of the current alias to avoid invalid placement of deallocs for aliases without any uses.
					Operation* beforeOp = endOperation;
					if (alias.getDefiningOp() &&
							!(beforeOp = placementBlock->findAncestorOpInBlock(
										*alias.getDefiningOp())))
						continue;

					Operation* aliasEndOperation =
							livenessInfo->getEndOperation(alias, beforeOp);

					// Check whether the aliasEndOperation lies in the desired block and
					// whether it is behind the current endOperation. If yes, this will be
					// the new endOperation.

					if (aliasEndOperation->getBlock() == placementBlock &&
							endOperation->isBeforeInBlock(aliasEndOperation))
						endOperation = aliasEndOperation;
				}

				// endOperation is the last operation behind which we can safely store
				// the dealloc taking all potential aliases into account.

				// If there is an existing dealloc, move it to the right place.
				Operation* deallocOperation = std::get<1>(entry);
				if (deallocOperation)
				{
					deallocOperation->moveAfter(endOperation);
				}
				else
				{
					// If the Dealloc position is at the terminator operation of the
					// block, then the value should escape from a deallocation.

					Operation* nextOp = endOperation->getNextNode();

					if (!nextOp)
						continue;

					// If there is no dealloc node, insert one in the right place.
					OpBuilder builder(nextOp);
					builder.create<FreeOp>(alloc.getLoc(), alloc);
				}
			}
		}

		private:
		PostDominanceInfo postDominators;
	};
}

// TODO: run on function to enable multithreading
class BufferDeallocationPass : public mlir::PassWrapper<BufferDeallocationPass, mlir::OperationPass<mlir::ModuleOp>>
{
	public:
	void runOnOperation() override
	{
		getOperation().walk([](FunctionOp op) {
			llvm::errs() << "Current op for dealloc: " << op.getName() << "\n";
			op.dump();
			BufferDeallocation deallocation(op);
			deallocation.deallocate();
			op.dump();
		});
	}
};

std::unique_ptr<mlir::Pass> marco::codegen::createBufferDeallocationPass()
{
	return std::make_unique<BufferDeallocationPass>();
}
