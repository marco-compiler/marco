#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/mlirlowerer/ops/OpTrait.h>
#include <modelica/mlirlowerer/Type.h>

namespace modelica
{
	class AssignmentOp : public mlir::Op<AssignmentOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroResult, mlir::OpTrait::VariadicOperands> {
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Value source, mlir::Value destination);
		void print(mlir::OpAsmPrinter &p);
		mlir::Value source();
		mlir::Value destination();
	};
}
