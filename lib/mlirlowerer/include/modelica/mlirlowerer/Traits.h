#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/BlockAndValueMapping.h>

#include "Type.h"

namespace modelica::codegen
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
		};
	}

	class BreakableOp : public mlir::OpInterface<BreakableOp, detail::BreakableOpTraits>
	{
		public:
		using OpInterface<BreakableOp, detail::BreakableOpTraits>::OpInterface;

		template <typename ConcreteOp>
		struct BreakableOpTrait : public mlir::OpInterface<BreakableOp, detail::BreakableOpTraits>::Trait<ConcreteOp>
		{

		};

		template <typename ConcreteOp>
		struct Trait : public BreakableOpTrait<ConcreteOp> {};
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
		struct Trait : public ClassTrait<ConcreteOp> {};

		void getMembers(llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names)
		{
			getImpl()->getMembers(getOperation(), members, names);
		}
	};

	namespace detail
	{
		struct VectorizableOpTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				virtual bool isVectorized(mlir::Operation* op) const = 0;
				virtual unsigned int vectorizationRank(mlir::Operation* op) const = 0;
				virtual mlir::ValueRange getArgs(mlir::Operation* op) const = 0;
				virtual unsigned int getArgExpectedRank(mlir::Operation* op, unsigned int argIndex) const = 0;
				virtual mlir::ValueRange scalarize(mlir::Operation* op, mlir::OpBuilder& builder, mlir::ValueRange indexes) const = 0;
			};

			template <typename ConcreteOp>
			struct Model : public Concept
			{
				bool isVectorized(mlir::Operation* op) const override
				{
					return vectorizationRank(op) != 0;
				}

				unsigned int vectorizationRank(mlir::Operation* op) const override
				{
					llvm::SmallVector<long, 2> expectedRanks;
					llvm::SmallVector<long, 3> dimensions;

					if (getArgs(op).empty())
						return 0;

					for (auto& arg : llvm::enumerate(getArgs(op)))
					{
						mlir::Type argType = arg.value().getType();
						unsigned int argExpectedRank = getArgExpectedRank(op, arg.index());

						unsigned int argActualRank = argType.isa<ArrayType>() ?
						    argType.cast<ArrayType>().getRank() : 0;

						// Each argument must have a rank higher than the expected one
						// for the operation to be vectorized.
						if (argActualRank <= argExpectedRank)
							return 0;

						if (arg.index() == 0)
						{
							// If this is the first argument, then it will determine the
							// rank and dimensions of the result array, although the
							// dimensions can be also specialized by the other arguments
							// if initially unknown.

							for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
							{
								auto& dimension = argType.cast<ArrayType>().getShape()[arg.index()];
								dimensions.push_back(dimension);
							}
						}
						else
						{
							// The rank difference must match with the one given by the first
							// argument, independently from the dimensions sizes.
							if (argActualRank != argExpectedRank + dimensions.size())
								return 0;

							for (size_t i = 0; i < argActualRank - argExpectedRank; ++i)
							{
								auto& dimension = argType.cast<ArrayType>().getShape()[i];

								// If the dimension is dynamic, then no further checks or
								// specializations are possible.
								if (dimension == -1)
									continue;

								// If the dimension determined by the first argument is fixed,
								// then also the dimension of the other arguments must match
								// (when that's fixed too).
								if (dimensions[i] != -1 && dimensions[i] != dimension)
									return 0;

								// If the dimension determined by the first argument is dynamic, then
								// set it to a required size.
								if (dimensions[i] == -1)
									dimensions[i] = dimension;
							}
						}
					}

					return dimensions.size();
				}

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

				bool isVectorized(mlir::Operation* op) const override
				{
					return false;
				}

				unsigned int vectorizationRank(mlir::Operation* op) const override
				{
					return 0;
				}

				mlir::ValueRange getArgs(mlir::Operation* op) const override
				{
					return op->getOperands();
				}

				unsigned int getArgExpectedRank(mlir::Operation* op, unsigned int argIndex) const override
				{
					return 0;
				}

				mlir::ValueRange scalarize(mlir::Operation* op, mlir::OpBuilder& builder, mlir::ValueRange indexes) const override
				{
					return mlir::cast<ConcreteOp>(op).scalarize(indexes);
				}
			};
		};
	}

	class VectorizableOpInterface : public mlir::OpInterface<VectorizableOpInterface, detail::VectorizableOpTraits>
	{
		public:
		using OpInterface<VectorizableOpInterface, detail::VectorizableOpTraits>::OpInterface;

		template <typename ConcreteOp>
		struct VectorizableOpTrait : public mlir::OpInterface<VectorizableOpInterface, detail::VectorizableOpTraits>::Trait<ConcreteOp>
		{
			unsigned int vectorizationRank()
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				return mlir::cast<VectorizableOpInterface>(op).vectorizationRank();
			}
		};

		template <typename ConcreteOp>
		struct Trait : public VectorizableOpTrait<ConcreteOp> {};

		bool isVectorized()
		{
			return getImpl()->isVectorized(getOperation());
		}

		unsigned int vectorizationRank()
		{
			return getImpl()->vectorizationRank(getOperation());
		}

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
		struct InvertibleInterfaceTraits
		{
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				// TODO: keep ValueRange or switch to Value?
				virtual mlir::LogicalResult invert(mlir::Operation* op, mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult) const = 0;
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
		};
	}

	class InvertibleInterface : public mlir::OpInterface<InvertibleInterface, detail::InvertibleInterfaceTraits>
	{
		public:
		using OpInterface<InvertibleInterface, detail::InvertibleInterfaceTraits>::OpInterface;

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
		};
	}

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
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::Value distributeNegateOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeNegateOp(builder, resultType);
				}
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
			class FallbackModel : public Concept
			{
				public:
				FallbackModel() = default;

				mlir::Value distributeMulOp(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value) const final
				{
					return mlir::cast<ConcreteOp>(op).distributeMulOp(builder, resultType, value);
				}
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
			struct Concept
			{
				Concept() = default;
				Concept(const Concept& other) = default;
				Concept(Concept&& other) = default;
				Concept& operator=(Concept&& other) = default;
				virtual ~Concept() = default;
				Concept& operator=(const Concept& other) = default;

				virtual mlir::ValueRange derive(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const = 0;
				virtual void getOperandsToBeDerived(mlir::Operation* op, llvm::SmallVectorImpl<mlir::Value>& toBeDerived) const = 0;
				//virtual mlir::ValueRange deriveTree(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const = 0;
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

				/*
				mlir::ValueRange deriveTree(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const final
				{
					return mlir::cast<ConcreteOp>(op).deriveTree(builder, derivatives);
				}
				 */
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

				/*
				mlir::ValueRange deriveTree(mlir::Operation* op, mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives) const final
				{
					return mlir::cast<ConcreteOp>(op).deriveTree(builder, derivatives);
				}
				 */
			};
		};
	}

	class DerivativeInterface : public mlir::OpInterface<DerivativeInterface, detail::DerivativeInterfaceTraits>
	{
		public:
		using OpInterface<DerivativeInterface, detail::DerivativeInterfaceTraits>::OpInterface;

		template <typename ConcreteOp>
		struct DerivativeTrait : public mlir::OpInterface<DerivativeInterface, detail::DerivativeInterfaceTraits>::Trait<ConcreteOp>
		{
			/*
			mlir::ValueRange deriveTree(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
			{
				mlir::Operation* op = (*static_cast<ConcreteOp*>(this)).getOperation();
				return mlir::cast<DerivativeInterface>(op).derive(builder, derivatives);
			}
			 */
		};

		template <typename ConcreteOp>
		struct Trait : public DerivativeTrait<ConcreteOp> {};

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
		{
			mlir::OpBuilder::InsertionGuard guard(builder);

			// The derivative is placed before the old assignment, in order to avoid
			// inconsistencies in case of self-assignments (i.e. "y := y * 2" would
			// invalidate the derivative if placed before "y' := y' * 2").
			builder.setInsertionPoint(getOperation());

			mlir::ValueRange ders = getImpl()->derive(getOperation(), builder, derivatives);

			for (const auto& [base, derived] : llvm::zip(getOperation()->getResults(), ders))
				derivatives.map(base, derived);

			return ders;
		}

		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
		{
			getImpl()->getOperandsToBeDerived(getOperation(), toBeDerived);
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
}
