#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/mlirlowerer/ops/OpTrait.h>

namespace modelica
{
	class CastOp : public mlir::Op<CastOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult> {
		public:
		using Op::Op;

		static ::llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Value value, mlir::Type destinationType);
		void print(mlir::OpAsmPrinter &p);
		mlir::Value value();
	};

	class CastCommonOp : public mlir::Op<CastCommonOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::VariadicResults> {
		public:
		using Op::Op;

		static ::llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::ValueRange values);
		void print(mlir::OpAsmPrinter &p);
		mlir::ValueRange values();
		mlir::Type type();
	};

	class AssignmentOp : public mlir::Op<AssignmentOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroResult, mlir::OpTrait::VariadicOperands> {
		public:
		using Op::Op;

		static ::llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Value source, mlir::Value destination);
		void print(mlir::OpAsmPrinter &p);
		mlir::Value source();
		mlir::Value destination();
	};
}
