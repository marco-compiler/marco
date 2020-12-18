#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/Type.hpp>

namespace modelica
{
	class AddOp : public mlir::Op<AddOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands);
	};

	class SubOp : public mlir::Op<AddOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands);
	};

	class MulOp : public mlir::Op<AddOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands);
	};

	class DivOp : public mlir::Op<AddOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::ValueRange operands);
	};
}
