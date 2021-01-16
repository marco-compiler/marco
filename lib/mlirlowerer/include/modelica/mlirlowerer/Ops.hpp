#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <modelica/frontend/Operation.hpp>
#include <modelica/frontend/Type.hpp>
#include <modelica/mlirlowerer/OpTrait.hpp>

namespace modelica
{
	class ArrayCopyOp : public mlir::Op<ArrayCopyOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::SameTypeOperands> {
		public:
		using Op::Op;

		static ::llvm::StringRef getOperationName();
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value source, mlir::Value destination);
		void print(::mlir::OpAsmPrinter &p);
	};

	class NegateOp : public mlir::Op<NegateOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType, mlir::OpTrait::IsInvolution>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value operand);
		void print(mlir::OpAsmPrinter& printer);
	};

	class AddOp : public mlir::Op<AddOp,mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};

	class SubOp : public mlir::Op<SubOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};

	class MulOp : public mlir::Op<MulOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};

	class DivOp : public mlir::Op<DivOp, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameOperandsAndResultType>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};

	class EqOp : public mlir::Op<EqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

	class NotEqOp : public mlir::Op<NotEqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

	class GtOp : public mlir::Op<GtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

	class GteOp : public mlir::Op<GteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

	class LtOp : public mlir::Op<LtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

	class LteOp : public mlir::Op<LteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
	};

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

	class ForOp : public mlir::Op<ForOp, mlir::OpTrait::NRegions<4>::Impl, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, BreakableLoop::Trait>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Region& condition();
		mlir::Region& step();
		mlir::Region& body();
		mlir::Region& exit();
	};

	class WhileOp : public mlir::Op<WhileOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, BreakableLoop::Trait>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Region& condition();
		mlir::Region& body();
		mlir::Region& exit();
	};

	class ConditionOp : public mlir::Op<ConditionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::HasParent<ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator> {
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value condition);
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value condition, mlir::ValueRange args);
		void print(::mlir::OpAsmPrinter &p);
		mlir::Value condition();
	};

	class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<IfOp, ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange operands);
		void print(mlir::OpAsmPrinter& printer);
	};

	class BreakOp : public mlir::Op<BreakOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::OneSuccessor, mlir::OpTrait::HasParent<IfOp, ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Block* successor);
		void print(mlir::OpAsmPrinter& printer);
	};
}
