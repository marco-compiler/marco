#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/Type.hpp>

namespace modelica
{
	/**
	 * This class verifies that all operands of the specified operation have a
	 * signless integer, float or index type, a vector thereof, or a tensor
	 * thereof.
	 */
	template <typename ConcreteType>
	class OperandsAreSignlessIntegerOrFloatLike
			: public mlir::OpTrait::TraitBase<ConcreteType, OperandsAreSignlessIntegerOrFloatLike> {
		public:
		static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
			if (failed(mlir::OpTrait::impl::verifyOperandsAreSignlessIntegerLike(op)))
				return mlir::OpTrait::impl::verifyOperandsAreFloatLike(op);

			return mlir::success();
		}
	};

	class NegateOp : public mlir::Op<NegateOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType, mlir::OpTrait::IsInvolution>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand);
	};

	class AddOp : public mlir::Op<AddOp,mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
	};

	class SubOp : public mlir::Op<SubOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
	};

	class MulOp : public mlir::Op<MulOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
	};

	class DivOp : public mlir::Op<DivOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
	};

	class EqOp : public mlir::Op<EqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};

	class NotEqOp : public mlir::Op<NotEqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};

	class GtOp : public mlir::Op<GtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};

	class GteOp : public mlir::Op<GteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};

	class LtOp : public mlir::Op<LtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};

	class LteOp : public mlir::Op<LteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
	};
}
