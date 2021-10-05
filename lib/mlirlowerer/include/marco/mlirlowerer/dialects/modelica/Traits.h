#pragma once

#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BlockAndValueMapping.h>

#include "Type.h"

namespace marco::codegen::modelica
{
	namespace detail
	{
		struct BreakableOpTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{

			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class BreakableOp : public mlir::OpInterface<BreakableOp, detail::BreakableOpTraits>
	{
		public:
		using OpInterface<BreakableOp, detail::BreakableOpTraits>::OpInterface;
	};

	namespace detail
	{
		struct ClassTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Get the members of the class.
				 *
				 * @param op				operation
				 * @param members 	members list to be populated
				 * @param names			names list to be populated
				 */
				virtual void getMembers(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				void getMembers(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names) const override
				{
					mlir::cast<ConcreteOp>(op).getMembers(members, names);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				void getMembers(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names) const override
				{

				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class ClassInterface : public mlir::OpInterface<ClassInterface, detail::ClassTraits>
	{
		public:
		using OpInterface<ClassInterface, detail::ClassTraits>::OpInterface;

		template <typename ConcreteOp>
		struct ClassTrait : public mlir::OpInterface<ClassInterface, detail::ClassTraits>::Trait<ConcreteOp>
		{
			void getMembers(llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names)
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				mlir::cast<ClassInterface>(op).getMembers(members, names);
			}
		};

		template <typename ConcreteOp>
		struct Trait : public ClassTrait<ConcreteOp>
		{

		};

		void getMembers(llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names)
		{
			getImpl()->getMembers(getOperation(), members, names);
		}
	};

	namespace detail
	{
		struct VectorizableOpTraits
		{
			struct Concept {
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Get the arguments of the operation.
				 *
				 * @param op 	operation
				 * @return arguments
				 */
				virtual mlir::ValueRange getArgs(mlir::Operation* op) const = 0;

				/**
				 * Get the expected rank of an argument in case of a scalar usage of
				 * the operation.
				 *
				 * @param op 				operation
				 * @param argIndex 	argument index
				 * @return expected rank
				 */
				virtual unsigned int getArgExpectedRank(mlir::Operation* op, unsigned int argIndex) const = 0;

				/**
				 * Convert the vectorized operation into a scalar one.
				 *
				 * @param op				vectorized operation
				 * @param builder 	operation builder
				 * @param indexes 	indexes upon which to iterate
				 * @return new results of the vectorized operation
				 */
				virtual mlir::ValueRange scalarize(mlir::Operation* op, mlir::OpBuilder& builder, mlir::ValueRange indexes) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::ValueRange getArgs(mlir::Operation* op) const override
				{
					return mlir::cast<ConcreteOp>(op).getArgs();
				}

				unsigned int getArgExpectedRank(mlir::Operation* op, unsigned int argIndex) const override
				{
					return mlir::cast<ConcreteOp>(op).getArgExpectedRank(argIndex);
				}

				mlir::ValueRange scalarize(mlir::Operation* op, mlir::OpBuilder& builder, mlir::ValueRange indexes) const override
				{
					return mlir::cast<ConcreteOp>(op).scalarize(builder, indexes);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::ValueRange getArgs(mlir::Operation* op) const override
				{
					// Consider all the operands
					return op->getOperands();
				}

				unsigned int getArgExpectedRank(mlir::Operation* op, unsigned int argIndex) const override
				{
					// The operands are expected to be scalars
					return 0;
				}

				mlir::ValueRange scalarize(mlir::Operation* op, mlir::OpBuilder& builder, mlir::ValueRange indexes) const override
				{
					// Don't touch the operation
					return op->getResults();
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class VectorizableOpInterface : public mlir::OpInterface<VectorizableOpInterface, detail::VectorizableOpTraits>
	{
		public:
		using OpInterface<VectorizableOpInterface, detail::VectorizableOpTraits>::OpInterface;

		mlir::ValueRange getArgs()
		{
			return getImpl()->getArgs(getOperation());
		}

		unsigned int getArgExpectedRank(unsigned int argIndex)
		{
			return getImpl()->getArgExpectedRank(getOperation(), argIndex);
		}

		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
		{
			return getImpl()->scalarize(getOperation(), builder, indexes);
		}
	};

	namespace detail
	{
		struct EquationInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				virtual mlir::Block* body(mlir::Operation* op) const = 0;
				virtual mlir::ValueRange inductions(mlir::Operation* op) const = 0;
				virtual mlir::Value induction(mlir::Operation* op, size_t index) const = 0;
				virtual long inductionIndex(mlir::Operation* op, mlir::Value induction) const = 0;
				virtual mlir::ValueRange lhs(mlir::Operation* op) const = 0;
				virtual mlir::ValueRange rhs(mlir::Operation* op) const = 0;
			};

			template<typename ConcreteOp>
			struct Model : public Concept
			{
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

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

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

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class EquationInterface : public mlir::OpInterface<EquationInterface, detail::EquationInterfaceTraits>
	{
		public:
		using mlir::OpInterface<EquationInterface, detail::EquationInterfaceTraits>::OpInterface;

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

	namespace detail
	{
		struct InvertibleOpInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Invert the operation with respect to one of its arguments.
				 *
				 * @param op							operation to be inverted
				 * @param builder 				operation builder
				 * @param argumentIndex 	index of the operand with respect to invert
				 * @param currentResult 	current operation results
				 * @return success status
				 */
				// TODO: keep ValueRange or switch to Value?
				virtual mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResults) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult) const final
				{
					return mlir::cast<ConcreteOp>(op).invert(builder, argumentIndex, currentResult);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult) const final
				{
					return mlir::cast<ConcreteOp>(op).invert(builder, argumentIndex, currentResult);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class InvertibleOpInterface : public mlir::OpInterface<InvertibleOpInterface, detail::InvertibleOpInterfaceTraits>
	{
		public:
		using OpInterface<InvertibleOpInterface, detail::InvertibleOpInterfaceTraits>::OpInterface;

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
		{
			return getImpl()->invert(getOperation(), builder, argumentIndex, currentResult);
		}
	};

	namespace detail
	{
		struct DistributableInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Distribute the operation among its arguments, if possible.
				 *
				 * @param op				operation to be distributed
				 * @param builder 	operation builder
				 * @return value that will replace the previous operation result
				 */
				virtual mlir::Value distribute(mlir::Operation* op, mlir::OpBuilder& builder) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::Value distribute(mlir::Operation* op, mlir::OpBuilder& builder) const final
				{
					return mlir::cast<ConcreteOp>(op).distribute(builder);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::Value distribute(mlir::Operation* op, mlir::OpBuilder& builder) const final
				{
					return mlir::cast<ConcreteOp>(op).distribute(builder);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	/**
	 * This interface is used to abstract an operation that can be propagated
	 * down into the operations tree (i.e. towards the leaf values).
	 */
	class DistributableInterface : public mlir::OpInterface<DistributableInterface, detail::DistributableInterfaceTraits>
	{
		public:
		using OpInterface<DistributableInterface, detail::DistributableInterfaceTraits>::OpInterface;

		mlir::Value distribute(mlir::OpBuilder& builder)
		{
			return getImpl()->distribute(getOperation(), builder);
		}
	};

	namespace detail
	{
		struct NegateOpDistributionInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Distribute a negate operation among the arguments.
				 *
				 * @param op 					operation
				 * @param builder 		operation builder
				 * @param resultType	desired result type after propagation
				 * @return result after propagation
				 */
				virtual mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeNegateOp(builder, resultType);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept {
				public:
				FallbackModel() = default;

				mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeNegateOp(builder, resultType);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class NegateOpDistributionInterface : public mlir::OpInterface<NegateOpDistributionInterface, detail::NegateOpDistributionInterfaceTraits>
	{
		public:
		using OpInterface<NegateOpDistributionInterface, detail::NegateOpDistributionInterfaceTraits>::OpInterface;

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
		{
			return getImpl()->distributeNegateOp(getOperation(), builder, resultType);
		}
	};

	namespace detail
	{
		struct MulOpDistributionInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Distribute a multiplication among the arguments.
				 *
				 * @param op 					operation
				 * @param builder 		operation builder
				 * @param resultType	desired result type after propagation
				 * @return result after propagation
				 */
				virtual mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeMulOp(builder, resultType, value);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept {
				public:
				FallbackModel() = default;

				mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeMulOp(builder, resultType, value);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class MulOpDistributionInterface : public mlir::OpInterface<MulOpDistributionInterface, detail::MulOpDistributionInterfaceTraits>
	{
		public:
		using OpInterface<MulOpDistributionInterface, detail::MulOpDistributionInterfaceTraits>::OpInterface;

		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
		{
			return getImpl()->distributeMulOp(getOperation(), builder, resultType, value);
		}
	};

	namespace detail
	{
		struct DivOpDistributionInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				/**
				 * Distribute a division among the arguments.
				 *
				 * @param op 					operation
				 * @param builder 		operation builder
				 * @param resultType	desired result type after propagation
				 * @return result after propagation
				 */
				virtual mlir::Value distributeDivOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::Value distributeDivOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeDivOp(builder, resultType, value);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::Value distributeDivOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeDivOp(builder, resultType, value);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class DivOpDistributionInterface : public mlir::OpInterface<DivOpDistributionInterface, detail::DivOpDistributionInterfaceTraits>
	{
		public:
		using OpInterface<DivOpDistributionInterface, detail::DivOpDistributionInterfaceTraits>::OpInterface;

		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
		{
			return getImpl()->distributeDivOp(getOperation(), builder, resultType, value);
		}
	};

	namespace detail
	{
		struct DerivativeInterfaceTraits
		{
			struct Concept {
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				virtual mlir::ValueRange derive(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const = 0;
				virtual void getOperandsToBeDerived(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& toBeDerived) const = 0;
				virtual void getDerivableRegions(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Region*>& regions) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				mlir::ValueRange derive(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const final
				{
					return mlir::cast<ConcreteOp>(op).derive(builder, derivatives);
				}

				void getOperandsToBeDerived(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& toBeDerived) const final
				{
					return mlir::cast<ConcreteOp>(op).getOperandsToBeDerived(toBeDerived);
				}

				void getDerivableRegions(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Region*>& regions) const final
				{
					return mlir::cast<ConcreteOp>(op).getDerivableRegions(regions);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::ValueRange derive(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const final
				{
					return mlir::cast<ConcreteOp>(op).derive(builder, derivatives);
				}

				void getOperandsToBeDerived(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& toBeDerived) const final
				{
					return mlir::cast<ConcreteOp>(op).getOperandsToBeDerived(toBeDerived);
				}

				void getDerivableRegions(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Region*>& regions) const final
				{
					return mlir::cast<ConcreteOp>(op).getDerivableRegions(regions);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel> {

			};
		};
	}

	class DerivativeInterface : public mlir::OpInterface<DerivativeInterface, detail::DerivativeInterfaceTraits>
	{
		public:
		using OpInterface<DerivativeInterface, detail::DerivativeInterfaceTraits>::OpInterface;

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
		{
			mlir::OpBuilder::InsertionGuard guard(builder);

			builder.setInsertionPointAfter(getOperation());
			mlir::ValueRange ders = getImpl()->derive(getOperation(), builder, derivatives);

			if (!ders.empty())
				for (const auto& [base, derived] : llvm::zip(getOperation()->getResults(), ders))
					derivatives.map(base, derived);

			return ders;
		}

		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
		{
			getImpl()->getOperandsToBeDerived(getOperation(), toBeDerived);
		}

		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
		{
			getImpl()->getDerivableRegions(getOperation(), regions);
		}

		mlir::ValueRange deriveTree(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
		{
			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPoint(getOperation());

			llvm::SmallVector<mlir::Value, 3> toBeDerived;
			getOperandsToBeDerived(toBeDerived);

			for (mlir::Value operand : toBeDerived)
				if (!derivatives.contains(operand))
					return llvm::None;

			for (mlir::Value operand : toBeDerived)
			{
				mlir::Operation* definingOp = operand.getDefiningOp();

				if (definingOp == nullptr)
					continue;

				if (auto derivableOp = mlir::dyn_cast<DerivativeInterface>(definingOp))
					if (auto results = derivableOp.deriveTree(builder, derivatives);
							results.size() != derivableOp->getNumResults())
						return llvm::None;
			}

			return derive(builder, derivatives);
		}
	};

	namespace detail
	{
		struct HeapAllocatorTraits
		{
			static llvm::StringRef getAutoFreeAttrName()
			{
				return "auto_free";
			}

			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				static llvm::StringRef getAutoFreeAttrName()
				{
					return HeapAllocatorTraits::getAutoFreeAttrName();
				}

				virtual bool shouldBeFreed(mlir::Operation* op) const
				{
					llvm::StringRef attrName = getAutoFreeAttrName();
					return op->template getAttrOfType<mlir::BoolAttr>(attrName).getValue();
				}

				void setAsAutomaticallyFreed(mlir::Operation* op)
				{
					auto attr = mlir::BoolAttr::get(op->getContext(), true);
					op->setAttr(getAutoFreeAttrName(), attr);
				}

				void setAsManuallyFreed(mlir::Operation* op)
				{
					auto attr = mlir::BoolAttr::get(op->getContext(), false);
					op->setAttr(getAutoFreeAttrName(), attr);
				}
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				bool shouldBeFreed(mlir::Operation* op) const override
				{
					llvm::StringRef attrName = getAutoFreeAttrName();
					return op->template getAttrOfType<mlir::BoolAttr>(attrName).getValue();
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				bool shouldBeFreed(mlir::Operation* op) const override
				{
					return true;
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class HeapAllocator : public mlir::OpInterface<HeapAllocator, detail::HeapAllocatorTraits>
	{
		public:
		using OpInterface<HeapAllocator, detail::HeapAllocatorTraits>::OpInterface;

		template <typename ConcreteOp>
		struct HeapAllocatorTrait : public mlir::OpInterface<HeapAllocator, detail::HeapAllocatorTraits>::Trait<ConcreteOp>
		{
			static llvm::StringRef getAutoFreeAttrName()
			{
				return detail::HeapAllocatorTraits::getAutoFreeAttrName();
			}

			bool shouldBeFreed()
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				return mlir::cast<HeapAllocator>(op).shouldBeFreed();
			}

			void setAsAutomaticallyFreed()
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				return mlir::cast<HeapAllocator>(op).setAsAutomaticallyFreed();
			}

			void setAsManuallyFreed()
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				return mlir::cast<HeapAllocator>(op).setAsManuallyFreed();
			}
		};

		template <typename ConcreteOp>
		struct Trait : public HeapAllocatorTrait<ConcreteOp> {};

		static llvm::StringRef getAutoFreeAttrName()
		{
			return detail::HeapAllocatorTraits::getAutoFreeAttrName();
		}

		bool shouldBeFreed()
		{
			return getImpl()->shouldBeFreed(getOperation());
		}

		void setAsAutomaticallyFreed()
		{
			getImpl()->setAsAutomaticallyFreed(getOperation());
		}

		void setAsManuallyFreed()
		{
			getImpl()->setAsManuallyFreed(getOperation());
		}
	};

	namespace detail
	{
		struct FoldableOpInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				virtual void foldConstants(mlir::Operation* op, mlir::OpBuilder& builder) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				void foldConstants(mlir::Operation* op, mlir::OpBuilder& builder) const final
				{
					mlir::cast<ConcreteOp>(op).foldConstants(builder);
				}
			};

			template<typename ConcreteOp>
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				void foldConstants(mlir::Operation* op, mlir::OpBuilder& builder) const final
				{
					mlir::cast<ConcreteOp>(op).foldConstants(builder);
				}
			};

			template <typename ConcreteModel, typename ConcreteOp>
			struct ExternalModel : public FallbackModel<ConcreteModel>
			{

			};
		};
	}

	class FoldableOpInterface : public mlir::OpInterface<FoldableOpInterface, detail::FoldableOpInterfaceTraits>
	{
		public:
		using OpInterface<FoldableOpInterface, detail::FoldableOpInterfaceTraits>::OpInterface;

		void foldConstants(mlir::OpBuilder& builder)
		{
			getImpl()->foldConstants(getOperation(), builder);
		}
	};
}
