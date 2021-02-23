#pragma once

#include <mlir/IR/OpDefinition.h>
#include <modelica/mlirlowerer/ops/OpTrait.h>
#include <modelica/mlirlowerer/Type.h>

#include "ops/BasicOps.h"
#include "ops/MathOps.h"

namespace modelica
{
	//===----------------------------------------------------------------------===//
	// Modelica::AllocaOp
	//===----------------------------------------------------------------------===//

	class AllocaOp : public mlir::Op<AllocaOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, PointerType::Shape shape = {}, mlir::ValueRange dimensions = {});
		void print(mlir::OpAsmPrinter& printer);

		PointerType getPointerType();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AllocOp
	//===----------------------------------------------------------------------===//

	class AllocOp : public mlir::Op<AllocOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, PointerType::Shape shape = {}, mlir::ValueRange dimensions = {});
		void print(mlir::OpAsmPrinter& printer);

		PointerType getPointerType();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::FreeOp
	//===----------------------------------------------------------------------===//

	class FreeOp;

	class FreeOpAdaptor
	{
		public:
		FreeOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		FreeOpAdaptor(FreeOp& op);

		mlir::Value memory();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class FreeOp : public mlir::Op<FreeOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = FreeOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DimOp
	//===----------------------------------------------------------------------===//

	class DimOp;

	class DimOpAdaptor
	{
		public:
		DimOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		DimOpAdaptor(DimOp& op);

		mlir::Value memory();
		mlir::Value dimension();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class DimOp : public mlir::Op<DimOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = DimOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Value dimension);
		void print(mlir::OpAsmPrinter& printer);

		PointerType getPointerType();
		mlir::Value memory();
		mlir::Value dimension();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LoadOp
	//===----------------------------------------------------------------------===//

	class LoadOp;

	class LoadOpAdaptor
	{
		public:
		LoadOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		LoadOpAdaptor(LoadOp& op);

		mlir::Value memory();
		mlir::ValueRange indexes();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class LoadOp : public mlir::Op<LoadOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = LoadOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes = {});
		void print(mlir::OpAsmPrinter& printer);

		PointerType getPointerType();
		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::StoreOp
	//===----------------------------------------------------------------------===//

	class StoreOp;

	class StoreOpAdaptor
	{
		public:
		StoreOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		StoreOpAdaptor(StoreOp& op);

		mlir::Value value();
		mlir::Value memory();
		mlir::ValueRange indexes();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class StoreOp :public mlir::Op<StoreOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = StoreOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory, mlir::ValueRange indexes = {});
		void print(mlir::OpAsmPrinter& printer);

		PointerType getPointerType();
		mlir::Value value();
		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::IfOp
	//===----------------------------------------------------------------------===//

	class IfOp;

	class IfOpAdaptor
	{
		public:
		IfOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		IfOpAdaptor(IfOp& op);

		mlir::Value condition();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class IfOp : public mlir::Op<IfOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::OneOperand> {
		public:
		using Op::Op;
		using Adaptor = IfOpAdaptor;

		static ::llvm::StringRef getOperationName();
		static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value cond, bool withElseRegion = false);
		void print(::mlir::OpAsmPrinter &p);

		mlir::Value condition();
		mlir::Region& thenRegion();
		mlir::Region& elseRegion();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ForOp
	//===----------------------------------------------------------------------===//

	class ForOp;

	class ForOpAdaptor
	{
		public:
		ForOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		ForOpAdaptor(ForOp& op);

		mlir::Value breakCondition();
		mlir::Value returnCondition();
		mlir::ValueRange args();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class ForOp : public mlir::Op<ForOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = ForOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Region& condition();
		mlir::Region& step();
		mlir::Region& body();

		mlir::Value breakCondition();
		mlir::Value returnCondition();
		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::WhileOp
	//===----------------------------------------------------------------------===//

	class WhileOp;

	class WhileOpAdaptor
	{
		public:
		WhileOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		WhileOpAdaptor(WhileOp& op);

		mlir::Value breakCondition();
		mlir::Value returnCondition();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class WhileOp : public mlir::Op<WhileOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = WhileOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Region& condition();
		mlir::Region& body();

		mlir::Value breakCondition();
		mlir::Value returnCondition();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ConditionOp
	//===----------------------------------------------------------------------===//

	class ConditionOp;

	class ConditionOpAdaptor
	{
		public:
		ConditionOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		ConditionOpAdaptor(ConditionOp& op);

		mlir::Value condition();
		mlir::ValueRange args();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class ConditionOp : public mlir::Op<ConditionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::HasParent<ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator> {
		public:
		using Op::Op;
		using Adaptor = ConditionOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, mlir::Value condition, mlir::ValueRange args = {});
		void print(::mlir::OpAsmPrinter &p);

		mlir::Value condition();
		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::YieldOp
	//===----------------------------------------------------------------------===//

	class YieldOp;

	class YieldOpAdaptor
	{
		public:
		YieldOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		YieldOpAdaptor(YieldOp& op);

		mlir::ValueRange args();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<IfOp, ForOp, WhileOp>::Impl, mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = YieldOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args = {});
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CastOp
	//===----------------------------------------------------------------------===//

	class CastOp;

	class CastOpAdaptor
	{
		public:
		CastOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		CastOpAdaptor(CastOp& op);

		mlir::Value value();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class CastOp : public mlir::Op<CastOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult> {
		public:
		using Op::Op;
		using Adaptor = CastOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& Builder, mlir::OperationState& state, mlir::Value value, mlir::Type resultType);
		void print(mlir::OpAsmPrinter &p);

		mlir::Value value();
		mlir::Type resultType();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CastCommonOp
	//===----------------------------------------------------------------------===//

	class CastCommonOp;

	class CastCommonOpAdaptor
	{
		public:
		CastCommonOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		CastCommonOpAdaptor(CastCommonOp& op);

		mlir::ValueRange operands();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class CastCommonOp : public mlir::Op<CastCommonOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::VariadicResults> {
		public:
		using Op::Op;
		using Adaptor = CastCommonOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values);
		void print(mlir::OpAsmPrinter &p);

		mlir::Type resultType();
		mlir::ValueRange operands();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::NegateOp
	//===----------------------------------------------------------------------===//

	class NegateOp;

	class NegateOpAdaptor
	{
		public:
		NegateOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		NegateOpAdaptor(NegateOp& op);

		mlir::Value operand();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class NegateOp : public mlir::Op<NegateOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, mlir::OpTrait::SameOperandsAndResultType, mlir::OpTrait::IsInvolution>
	//class NegateOp : public mlir::Op<NegateOp, mlir::OpTrait::OneOperand, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = NegateOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState& state, mlir::Value operand);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::EqOp
	//===----------------------------------------------------------------------===//

	class EqOp;

	class EqOpAdaptor
	{
		public:
		EqOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		EqOpAdaptor(EqOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class EqOp : public mlir::Op<EqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;
		using Adaptor = EqOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::NotEqOp
	//===----------------------------------------------------------------------===//

	class NotEqOp;

	class NotEqOpAdaptor
	{
		public:
		NotEqOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		NotEqOpAdaptor(NotEqOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class NotEqOp : public mlir::Op<NotEqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;
		using Adaptor = NotEqOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::GtOp
	//===----------------------------------------------------------------------===//

	class GtOp;

	class GtOpAdaptor
	{
		public:
		GtOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		GtOpAdaptor(GtOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class GtOp : public mlir::Op<GtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;
		using Adaptor = GtOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::GteOp
	//===----------------------------------------------------------------------===//

	class GteOp;

	class GteOpAdaptor
	{
		public:
		GteOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		GteOpAdaptor(GteOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class GteOp : public mlir::Op<GteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;
		using Adaptor = GteOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LtOp
	//===----------------------------------------------------------------------===//

	class LtOp;

	class LtOpAdaptor
	{
		public:
		LtOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		LtOpAdaptor(LtOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class LtOp : public mlir::Op<LtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;
		using Adaptor = LtOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LteOp
	//===----------------------------------------------------------------------===//

	class LteOp;

	class LteOpAdaptor
	{
		public:
		LteOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		LteOpAdaptor(LteOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class LteOp : public mlir::Op<LteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, OperandsAreSignlessIntegerOrFloatLike, mlir::OpTrait::SameTypeOperands, mlir::OpTrait::ResultsAreBoolLike>
	{
		public:
		using Op::Op;
		using Adaptor = LteOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value lhs, mlir::Value rhs);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		mlir::LogicalResult verify();
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AddOp
	//===----------------------------------------------------------------------===//

	class AddOp;

	class AddOpAdaptor
	{
		public:
		AddOpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr);
		AddOpAdaptor(AddOp& op);

		mlir::Value lhs();
		mlir::Value rhs();

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	class AddOp : public mlir::Op<AddOp,mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::IsCommutative>
	{
		public:
		using Op::Op;
		using Adaptor = AddOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};
}
