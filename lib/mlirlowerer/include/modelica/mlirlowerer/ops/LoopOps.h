#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/mlirlowerer/ops/OpTrait.h>

namespace modelica
{
	class ConditionOp;
	class YieldOp;

	class IfOp : public mlir::Op<IfOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::OneOperand> {
		public:
		using Op::Op;

		static ::llvm::StringRef getOperationName();
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value cond, bool withElseRegion);
		void print(::mlir::OpAsmPrinter &p);
		mlir::Value condition();
		mlir::Region& thenRegion();
		mlir::Region& elseRegion();
	};

	class ForOp : public mlir::Op<ForOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Region& condition();
		mlir::Region& step();
		mlir::Region& body();
		mlir::Value breakCondition();
		mlir::Value returnCondition();
		mlir::Operation::operand_range args();
	};

	class WhileOp : public mlir::Op<WhileOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Region& condition();
		mlir::Region& body();
		mlir::Value breakCondition();
		mlir::Value returnCondition();
	};

	class ConditionOp : public mlir::Op<ConditionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::HasParent<ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator> {
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value condition);
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value condition, mlir::ValueRange args);
		void print(::mlir::OpAsmPrinter &p);
		mlir::Value condition();
		mlir::Operation::operand_range args();
	};

	class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};
}
