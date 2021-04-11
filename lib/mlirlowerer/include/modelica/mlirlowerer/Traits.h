#pragma once

#include <mlir/IR/OpDefinition.h>

struct EquationInterfaceTraits {
	// Define a base concept class that specifies the virtual interface
	// to be implemented.
	struct Concept {
		virtual mlir::Block* body(mlir::Operation* op) const = 0;
		virtual mlir::ValueRange inductions(mlir::Operation* op) const = 0;
		virtual mlir::Value induction(mlir::Operation* op, size_t index) const = 0;
		virtual long inductionIndex(mlir::Operation* op, mlir::Value induction) const = 0;
		virtual mlir::ValueRange lhs(mlir::Operation* op) const = 0;
		virtual mlir::ValueRange rhs(mlir::Operation* op) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::Block* body(mlir::Operation* op) const final
		{
			return mlir::cast<ConcreteOp>(op).body();
		}

		mlir::ValueRange inductions(mlir::Operation* op) const final
		{
			return mlir::cast<ConcreteOp>(op).inductions();
		}

		mlir::Value induction(mlir::Operation* op, size_t index) const final
		{
			return mlir::cast<ConcreteOp>(op).induction(index);
		}

		long inductionIndex(mlir::Operation* op, mlir::Value induction) const final
		{
			return mlir::cast<ConcreteOp>(op).inductionIndex(induction);
		}

		mlir::ValueRange lhs(mlir::Operation* op) const final
		{
			return mlir::cast<ConcreteOp>(op).lhs();
		}

		mlir::ValueRange rhs(mlir::Operation* op) const final
		{
			return mlir::cast<ConcreteOp>(op).rhs();
		}
	};
};

// Define the main interface class that analyses and transformations will
// interface with.
class EquationInterface : public mlir::OpInterface<EquationInterface, EquationInterfaceTraits> {
	public:
	// Inherit the base class constructor to support LLVM-style casting
	using OpInterface<EquationInterface, EquationInterfaceTraits>::OpInterface;

	mlir::Block* body()
	{
		return getImpl()->body(getOperation());
	}

	mlir::ValueRange inductions()
	{
		return getImpl()->inductions(getOperation());
	}

	mlir::Value induction(size_t index)
	{
		return getImpl()->induction(getOperation(), index);
	}

	long inductionIndex(mlir::Value induction)
	{
		return getImpl()->inductionIndex(getOperation(), induction);
	}

	mlir::ValueRange lhs()
	{
		return getImpl()->lhs(getOperation());
	}

	mlir::ValueRange rhs()
	{
		return getImpl()->rhs(getOperation());
	}
};
