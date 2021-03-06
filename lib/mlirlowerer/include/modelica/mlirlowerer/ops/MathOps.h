#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/mlirlowerer/ops/OpTrait.h>

namespace modelica
{
	class CrossProductOp : public mlir::Op<CrossProductOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Value lhs();
		mlir::Value rhs();
	};
}
