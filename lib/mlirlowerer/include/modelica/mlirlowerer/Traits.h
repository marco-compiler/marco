#pragma once

#include <mlir/IR/OpDefinition.h>

struct EquationInterfaceTraits {
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

class EquationInterface : public mlir::OpInterface<EquationInterface, EquationInterfaceTraits> {
	public:
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

struct InvertibleInterfaceTraits {
	struct Concept {
		// TODO: keep ValueRange or switch to Value?
		virtual mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult) const final
		{
			return mlir::cast<ConcreteOp>(op).invert(builder, argumentIndex, currentResult);
		}
	};
};

class InvertibleInterface : public mlir::OpInterface<InvertibleInterface, InvertibleInterfaceTraits> {
	public:
	using OpInterface<InvertibleInterface, InvertibleInterfaceTraits>::OpInterface;

	mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
	{
		return getImpl()->invert(getOperation(), builder, argumentIndex, currentResult);
	}
};

struct DistributableInterfaceTraits {
	struct Concept {
		virtual mlir::Value distribute(mlir::Operation* op, mlir::OpBuilder& builder) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::Value distribute(mlir::Operation* op, mlir::OpBuilder& builder) const final
		{
			return mlir::cast<ConcreteOp>(op).distribute(builder);
		}
	};
};

class DistributableInterface : public mlir::OpInterface<DistributableInterface, DistributableInterfaceTraits> {
	public:
	using OpInterface<DistributableInterface, DistributableInterfaceTraits>::OpInterface;

	mlir::Value distribute(mlir::OpBuilder& builder)
	{
		return getImpl()->distribute(getOperation(), builder);
	}
};

struct NegateOpDistributionInterfaceTraits {
	struct Concept {
		virtual mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const final
		{
			return mlir::cast<ConcreteOp>(op).distributeNegateOp(builder, resultType);
		}
	};
};

class NegateOpDistributionInterface : public mlir::OpInterface<NegateOpDistributionInterface, NegateOpDistributionInterfaceTraits> {
	public:
	using OpInterface<NegateOpDistributionInterface, NegateOpDistributionInterfaceTraits>::OpInterface;

	mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
	{
		return getImpl()->distributeNegateOp(getOperation(), builder, resultType);
	}
};

struct MulOpDistributionInterfaceTraits {
	struct Concept {
		virtual mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
		{
			return mlir::cast<ConcreteOp>(op).distributeMulOp(builder, resultType, value);
		}
	};
};

class MulOpDistributionInterface : public mlir::OpInterface<MulOpDistributionInterface, MulOpDistributionInterfaceTraits> {
	public:
	using OpInterface<MulOpDistributionInterface, MulOpDistributionInterfaceTraits>::OpInterface;

	mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
	{
		return getImpl()->distributeMulOp(getOperation(), builder, resultType, value);
	}
};

struct DivOpDistributionInterfaceTraits {
	struct Concept {
		virtual mlir::Value distributeDivOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const = 0;
	};

	template <typename ConcreteOp>
	struct Model : public Concept {
		mlir::Value distributeDivOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
		{
			return mlir::cast<ConcreteOp>(op).distributeDivOp(builder, resultType, value);
		}
	};
};

class DivOpDistributionInterface : public mlir::OpInterface<DivOpDistributionInterface, DivOpDistributionInterfaceTraits> {
	public:
	using OpInterface<DivOpDistributionInterface, DivOpDistributionInterfaceTraits>::OpInterface;

	mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
	{
		return getImpl()->distributeDivOp(getOperation(), builder, resultType, value);
	}
};
