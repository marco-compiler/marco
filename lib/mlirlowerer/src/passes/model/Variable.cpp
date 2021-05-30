#include <modelica/mlirlowerer/passes/model/Model.h>
#include <modelica/mlirlowerer/passes/model/Variable.h>
#include <modelica/mlirlowerer/passes/model/VectorAccess.h>
#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica::codegen;
using namespace modelica::codegen::model;

class Variable::Impl
{
	public:
	Impl(mlir::Value memory) : reference(memory), state(false)
	{
		mlir::Operation* op = memory.getDefiningOp();
		assert(mlir::isa<AllocaOp>(op) || mlir::isa<AllocOp>(op));

		if (auto allocaOp = mlir::dyn_cast<AllocaOp>(op))
			constant = allocaOp.isConstant();
		else if (auto allocOp = mlir::dyn_cast<AllocOp>(op))
			constant = allocOp.isConstant();
		else
			constant = false;
	}

	friend class Variable;

	private:
	mlir::Value reference;
	bool state;
	bool constant;
	mlir::Value der;
};

Variable::Variable(mlir::Value memory)
		: impl(std::make_shared<Impl>(memory))
{
}

bool Variable::operator==(const Variable& rhs) const
{
	return impl == rhs.impl;
}

bool Variable::operator!=(const Variable& rhs) const
{
	return !(rhs == *this);
}

bool Variable::operator<(const Variable& rhs) const
{
	return impl < rhs.impl;
}

bool Variable::operator>(const Variable& rhs) const
{
	return rhs < *this;
}

bool Variable::operator<=(const Variable& rhs) const
{
	return !(rhs < *this);
}

bool Variable::operator>=(const Variable& rhs) const
{
	return !(*this < rhs);
}

mlir::Value Variable::getReference()
{
	return impl->reference;
}

bool Variable::isState() const
{
	return impl->state || isTime();
}

bool Variable::isConstant() const
{
	return impl->constant;
}

bool Variable::isTime() const
{
	auto simulation = impl->reference.getParentRegion()->getParentOfType<SimulationOp>();
	auto initTerminator = mlir::cast<YieldOp>(simulation.init().front().getTerminator());
	return impl->reference == initTerminator.values()[0];
}

mlir::Value Variable::getDer()
{
	return impl->der;
}

void Variable::setDer(mlir::Value value)
{
	impl->state = true;
	impl->der = value;
}

modelica::IndexSet Variable::toIndexSet() const
{
	return IndexSet({ toMultiDimInterval() });
}

modelica::MultiDimInterval Variable::toMultiDimInterval() const
{
	llvm::SmallVector<Interval, 2> intervals;
	assert(impl->reference.getType().isa<PointerType>());
	auto pointerType = impl->reference.getType().cast<PointerType>();

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
	assert(impl->reference.getType().isa<PointerType>());
	auto pointerType = impl->reference.getType().cast<PointerType>();
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
