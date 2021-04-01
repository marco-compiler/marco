#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen::model;

Variable::Variable(mlir::Value memory)
		: reference(memory), state(false)
{
	mlir::Operation* op = memory.getDefiningOp();
	assert(mlir::isa<AllocaOp>(op) || mlir::isa<AllocOp>(op));

	if (auto allocaOp = mlir::dyn_cast<AllocaOp>(op))
		constant = allocaOp.isConstant();

	if (auto allocOp = mlir::dyn_cast<AllocOp>(op))
		constant = allocOp.isConstant();

	constant = false;
}

mlir::Value Variable::getReference()
{
	return reference;
}

bool Variable::isState() const
{
	return state;
}

bool Variable::isConstant() const
{
	return constant;
}

mlir::Value Variable::getDer()
{
	return der;
}

void Variable::setDer(mlir::Value value)
{
	state = true;
	der = value;
}

modelica::IndexSet Variable::toIndexSet() const
{
	return IndexSet({ toMultiDimInterval() });
}

modelica::MultiDimInterval Variable::toMultiDimInterval() const
{
	llvm::SmallVector<Interval, 2> intervals;
	assert(reference.getType().isa<PointerType>());
	auto pointerType = reference.getType().cast<PointerType>();

	if (pointerType.getRank() == 0)
		intervals.emplace_back(0, 1);
	else
	{
		for (auto size : pointerType.getShape())
		{
			assert(size != -1);
			intervals.emplace_back(0, size);
		}
	}

	return MultiDimInterval(std::move(intervals));
}

size_t Variable::indexOfElement(llvm::ArrayRef<size_t> access) const
{
	assert(reference.getType().isa<PointerType>());
	auto pointerType = reference.getType().cast<PointerType>();
	assert(access.size() == pointerType.getRank());

	auto shape = pointerType.getShape();

	size_t index = 0;
	size_t maxIndex = 1;

	for (size_t i = access.size() - 1; i != std::numeric_limits<size_t>::max(); --i)
	{
		index += access[i] * maxIndex;
		assert(shape[i] != -1);
		maxIndex *= shape[i];
	}

	return index;
}
