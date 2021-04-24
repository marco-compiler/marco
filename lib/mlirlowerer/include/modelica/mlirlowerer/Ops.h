#pragma once

#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/IR/OpDefinition.h>

#include "Traits.h"
#include "Type.h"

namespace modelica::codegen
{
	/**
	 * Generic operation adaptor.
	 *
	 * The purpose of an adaptor is to allow to access specific values in both
	 * the operation and the conversion pattern without relying on hard-coded
	 * constants in both places.
	 *
	 * @tparam OpType operation class
	 */
	template<typename OpType>
	class OpAdaptor
	{
		public:
		OpAdaptor(mlir::ValueRange values, mlir::DictionaryAttr attrs = nullptr)
				: values(values), attrs(attrs)
		{
		}

		OpAdaptor(OpType& op)
				: values(op->getOperands()), attrs(op->getAttrDictionary())
		{
		}

		protected:
		[[nodiscard]] mlir::ValueRange getValues() const
		{
			return values;
		}

		[[nodiscard]] mlir::DictionaryAttr getAttrs() const
		{
			return attrs;
		}

		private:
		mlir::ValueRange values;
		mlir::DictionaryAttr attrs;
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PackOp
	//===----------------------------------------------------------------------===//

	class PackOp;

	class PackOpAdaptor : public OpAdaptor<PackOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange values();
	};

	class PackOp : public mlir::Op<PackOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = PackOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values);
		void print(mlir::OpAsmPrinter& printer);

		StructType resultType();
		mlir::ValueRange values();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ExtractOp
	//===----------------------------------------------------------------------===//

	class ExtractOp;

	class ExtractOpAdaptor : public OpAdaptor<ExtractOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value packedValue();
	};

	class ExtractOp : public mlir::Op<ExtractOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = ExtractOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value packedValue, unsigned int index);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value packedValue();
		unsigned int index();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SimulationOp
	//===----------------------------------------------------------------------===//

	class SimulationOp;

	class SimulationOpAdaptor : public OpAdaptor<SimulationOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class SimulationOp : public mlir::Op<SimulationOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, mlir::RegionBranchOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SimulationOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, RealAttribute startTime, RealAttribute endTime, RealAttribute timeStep, mlir::TypeRange vars);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		RealAttribute startTime();
		RealAttribute endTime();
		RealAttribute timeStep();

		mlir::Region& init();
		mlir::Region& body();
		mlir::Region& print();

		mlir::Value getVariableAllocation(mlir::Value var);

		mlir::Value time();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::EquationOp
	//===----------------------------------------------------------------------===//

	class EquationOp;

	class EquationOpAdaptor : public OpAdaptor<EquationOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class EquationOp : public mlir::Op<EquationOp, mlir::OpTrait::OneRegion, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, EquationInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = EquationOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Block* body();
		mlir::ValueRange inductions();
		mlir::Value induction(size_t index);
		long inductionIndex(mlir::Value induction);

		mlir::ValueRange lhs();
		mlir::ValueRange rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ForEquationOp
	//===----------------------------------------------------------------------===//

	class ForEquationOp;

	class ForEquationOpAdaptor : public OpAdaptor<ForEquationOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class ForEquationOp : public mlir::Op<ForEquationOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult, EquationInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ForEquationOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, size_t inductionsAmount);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Block* inductionsBlock();
		mlir::ValueRange inductionsDefinitions();

		mlir::Block* body();
		mlir::ValueRange inductions();
		mlir::Value induction(size_t index);
		long inductionIndex(mlir::Value induction);

		mlir::ValueRange lhs();
		mlir::ValueRange rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::InductionOp
	//===----------------------------------------------------------------------===//

	class InductionOp;

	class InductionOppAdaptor : public OpAdaptor<InductionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class InductionOp : public mlir::Op<InductionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult, mlir::OpTrait::HasParent<ForEquationOp>::Impl>
	{
		public:
		using Op::Op;
		using Adaptor = InductionOppAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, long start, long end);
		void print(mlir::OpAsmPrinter& printer);

		long start();
		long end();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::EquationSidesOp
	//===----------------------------------------------------------------------===//

	class EquationSidesOp;

	class EquationSidesOpAdaptor : public OpAdaptor<EquationSidesOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange lhs();
		mlir::ValueRange rhs();
	};

	class EquationSidesOp : public mlir::Op<EquationSidesOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<EquationOp, ForEquationOp>::Impl, mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = EquationSidesOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange lhs, mlir::ValueRange rhs);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange lhs();
		mlir::ValueRange rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ConstantOp
	//===----------------------------------------------------------------------===//

	class ConstantOp;

	class ConstantOpAdaptor : public OpAdaptor<ConstantOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class ConstantOp : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneResult, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ConstantLike> {
		public:
		using Op::Op;
		using Adaptor = ConstantOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute value);
		void print(mlir::OpAsmPrinter& printer);
		mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> operands);

		mlir::Attribute value();
		mlir::Type resultType();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CastOp
	//===----------------------------------------------------------------------===//

	class CastOp;

	class CastOpAdaptor : public OpAdaptor<CastOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value value();
	};

	class CastOp : public mlir::Op<CastOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult> {
		public:
		using Op::Op;
		using Adaptor = CastOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& Builder, mlir::OperationState& state, mlir::Value value, mlir::Type resultType);
		void print(mlir::OpAsmPrinter &p);
		mlir::LogicalResult verify();
		mlir::OpFoldResult fold(mlir::ArrayRef<mlir::Attribute> operands);

		mlir::Value value();
		mlir::Type resultType();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CastCommonOp
	//===----------------------------------------------------------------------===//

	class CastCommonOp;

	class CastCommonOpAdaptor : public OpAdaptor<CastCommonOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange operands();
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
	// Modelica::AssignmentOp
	//===----------------------------------------------------------------------===//

	class AssignmentOp;

	class AssignmentOpAdaptor : public OpAdaptor<AssignmentOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value source();
		mlir::Value destination();
	};

	class AssignmentOp : public mlir::Op<AssignmentOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::ZeroResult, mlir::OpTrait::VariadicOperands, mlir::MemoryEffectOpInterface::Trait> {
		public:
		using Op::Op;
		using Adaptor = AssignmentOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::Value destination);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value source();
		mlir::Value destination();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CallOp
	//===----------------------------------------------------------------------===//

	class CallOp;

	class CallOpAdaptor : public OpAdaptor<CallOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange args();
	};

	class CallOp : public mlir::Op<CallOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicResults, mlir::OpTrait::VariadicOperands, mlir::MemoryEffectOpInterface::Trait, mlir::CallOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = CallOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::TypeRange results, mlir::ValueRange args, unsigned int movedResults = 0);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);
		mlir::CallInterfaceCallable getCallableForCallee();
		mlir::Operation::operand_range getArgOperands();

		mlir::StringRef callee();
		mlir::ValueRange args();
		unsigned int movedResults();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AllocaOp
	//===----------------------------------------------------------------------===//

	class AllocaOp;

	class AllocaOpAdaptor : public OpAdaptor<AllocaOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange dynamicDimensions();
	};

	class AllocaOp : public mlir::Op<AllocaOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AllocaOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape = {}, mlir::ValueRange dimensions = {}, bool constant = false);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		PointerType resultType();
		mlir::ValueRange dynamicDimensions();
		bool isConstant();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AllocOp
	//===----------------------------------------------------------------------===//

	class AllocOp;

	class AllocOpAdaptor : public OpAdaptor<AllocOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange dynamicDimensions();
	};

	class AllocOp : public mlir::Op<AllocOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AllocOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape = {}, mlir::ValueRange dimensions = {}, bool shouldBeFreed = true, bool isConstant = false);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		bool shouldBeFreed();
		PointerType resultType();
		mlir::ValueRange dynamicDimensions();
		bool isConstant();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::FreeOp
	//===----------------------------------------------------------------------===//

	class FreeOp;

	class FreeOpAdaptor : public OpAdaptor<FreeOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
	};

	class FreeOp : public mlir::Op<FreeOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::ZeroResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = FreeOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PtrCastOp
	//===----------------------------------------------------------------------===//

	/**
	 * This operation should be used only for two purposes: the first one is
	 * for function calls, to remove the allocation scope before passing the
	 * array pointers as arguments to the functions, or to generalize the sizes
	 * to unknown ones; the second one is to cast from / to opaque pointers.
	 * The operation is NOT intended to be used to change the allocation scope
	 * (i.e. stack -> heap or heap -> stack), to cast the element type to a
	 * different one or to to specialize the shape to a fixed one. This last
	 * case is a certain sense violated by the opaque pointer casting scenario,
	 * and must be carefully checked by the user.
	 */
	class PtrCastOp;

	class PtrCastOpAdaptor : public OpAdaptor<PtrCastOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
	};

	class PtrCastOp : public mlir::Op<PtrCastOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = PtrCastOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value, mlir::Type resultType);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DimOp
	//===----------------------------------------------------------------------===//

	class DimOp;

	class DimOpAdaptor : public OpAdaptor<DimOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
		mlir::Value dimension();
	};

	class DimOp : public mlir::Op<DimOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = DimOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Value dimension);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		mlir::OpFoldResult fold(mlir::ArrayRef<mlir::Attribute> operands);

		PointerType getPointerType();
		mlir::Value memory();
		mlir::Value dimension();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SubscriptionOp
	//===----------------------------------------------------------------------===//

	class SubscriptionOp;

	class SubscriptionOpAdaptor : public OpAdaptor<SubscriptionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value source();
		mlir::ValueRange indexes();
	};

	class SubscriptionOp : public mlir::Op<SubscriptionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::ViewLikeOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SubscriptionOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange indexes);
		void print(mlir::OpAsmPrinter& printer);
		mlir::Value getViewSource();

		PointerType resultType();
		mlir::Value source();
		mlir::ValueRange indexes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LoadOp
	//===----------------------------------------------------------------------===//

	class LoadOp;

	class LoadOpAdaptor : public OpAdaptor<LoadOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	class LoadOp : public mlir::Op<LoadOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::AtLeastNOperands<1>::Impl, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = LoadOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes = {});
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		PointerType getPointerType();
		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::StoreOp
	//===----------------------------------------------------------------------===//

	class StoreOp;

	class StoreOpAdaptor : public OpAdaptor<StoreOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value value();
		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	class StoreOp :public mlir::Op<StoreOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::ZeroResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = StoreOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory, mlir::ValueRange indexes = {});
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		PointerType getPointerType();
		mlir::Value value();
		mlir::Value memory();
		mlir::ValueRange indexes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ArrayCloneOp
	//===----------------------------------------------------------------------===//

	class ArrayCloneOp;

	class ArrayCloneOpAdaptor : public OpAdaptor<ArrayCloneOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value source();
	};

	class ArrayCloneOp :public mlir::Op<ArrayCloneOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ArrayCloneOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, PointerType resultType, bool shouldBeFreed = true);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		bool shouldBeFreed();
		PointerType resultType();
		mlir::Value source();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::IfOp
	//===----------------------------------------------------------------------===//

	class IfOp;
	class YieldOp;

	class IfOpAdaptor : public OpAdaptor<IfOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value condition();
	};

	class IfOp : public mlir::Op<IfOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::VariadicResults, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::OneOperand, mlir::RegionBranchOpInterface::Trait, mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
		public:
		using Op::Op;
		using Adaptor = IfOpAdaptor;

		static ::llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::TypeRange resultTypes, mlir::Value cond, bool withElseRegion = false);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value cond, bool withElseRegion = false);
		void print(::mlir::OpAsmPrinter &p);
		mlir::LogicalResult verify();
		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::TypeRange resultTypes();
		mlir::Value condition();
		mlir::Region& thenRegion();
		mlir::Region& elseRegion();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ForOp
	//===----------------------------------------------------------------------===//

	class ForOp;

	class ForOpAdaptor : public OpAdaptor<ForOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange args();
	};

	class ForOp : public mlir::Op<ForOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::RegionBranchOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ForOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args = {});
		void print(mlir::OpAsmPrinter& printer);
		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::Region& condition();
		mlir::Region& step();
		mlir::Region& body();

		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::BreakableForOp
	//===----------------------------------------------------------------------===//

	class BreakableForOp;

	class BreakableForOpAdaptor : public OpAdaptor<BreakableForOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value breakCondition();
		mlir::Value returnCondition();
		mlir::ValueRange args();
	};

	class BreakableForOp : public mlir::Op<BreakableForOp, mlir::OpTrait::NRegions<3>::Impl, mlir::OpTrait::AtLeastNOperands<2>::Impl, mlir::OpTrait::ZeroResult, mlir::RegionBranchOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = BreakableForOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition, mlir::ValueRange args = {});
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::Region& condition();
		mlir::Region& step();
		mlir::Region& body();

		mlir::Value breakCondition();
		mlir::Value returnCondition();
		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::BreakableWhileOp
	//===----------------------------------------------------------------------===//

	class BreakableWhileOp;

	class BreakableWhileOpAdaptor : public OpAdaptor<BreakableWhileOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value breakCondition();
		mlir::Value returnCondition();
	};

	class BreakableWhileOp : public mlir::Op<BreakableWhileOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::ZeroResult, mlir::RegionBranchOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = BreakableWhileOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value breakCondition, mlir::Value returnCondition);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::Region& condition();
		mlir::Region& body();

		mlir::Value breakCondition();
		mlir::Value returnCondition();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ConditionOp
	//===----------------------------------------------------------------------===//

	class ConditionOp;

	class ConditionOpAdaptor : public OpAdaptor<ConditionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value condition();
		mlir::ValueRange args();
	};

	class ConditionOp : public mlir::Op<ConditionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::HasParent<ForOp, BreakableForOp, BreakableWhileOp>::Impl, mlir::OpTrait::IsTerminator> {
		public:
		using Op::Op;
		using Adaptor = ConditionOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args = {});
		void print(::mlir::OpAsmPrinter &p);
		mlir::LogicalResult verify();

		mlir::Value condition();
		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::YieldOp
	//===----------------------------------------------------------------------===//

	class YieldOp;

	class YieldOpAdaptor : public OpAdaptor<YieldOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange args();
	};

	class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::HasParent<ForEquationOp, IfOp, ForOp, BreakableForOp, BreakableWhileOp, SimulationOp>::Impl, mlir::OpTrait::IsTerminator>
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
	// Modelica::NotOp
	//===----------------------------------------------------------------------===//

	class NotOp;

	class NotOpAdaptor : public OpAdaptor<NotOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class NotOp : public mlir::Op<NotOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = NotOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AndOp
	//===----------------------------------------------------------------------===//

	class AndOp;

	class AndOpAdaptor : public OpAdaptor<AndOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class AndOp : public mlir::Op<AndOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::IsCommutative, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AndOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::OrOp
	//===----------------------------------------------------------------------===//

	class OrOp;

	class OrOpAdaptor : public OpAdaptor<OrOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class OrOp : public mlir::Op<OrOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::IsCommutative, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = OrOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();

		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::EqOp
	//===----------------------------------------------------------------------===//

	class EqOp;

	class EqOpAdaptor : public OpAdaptor<EqOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class EqOp : public mlir::Op<EqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::IsCommutative, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = EqOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::NotEqOp
	//===----------------------------------------------------------------------===//

	class NotEqOp;

	class NotEqOpAdaptor : public OpAdaptor<NotEqOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class NotEqOp : public mlir::Op<NotEqOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::IsCommutative, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = NotEqOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::GtOp
	//===----------------------------------------------------------------------===//

	class GtOp;

	class GtOpAdaptor : public OpAdaptor<GtOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class GtOp : public mlir::Op<GtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = GtOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::GteOp
	//===----------------------------------------------------------------------===//

	class GteOp;

	class GteOpAdaptor : public OpAdaptor<GteOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class GteOp : public mlir::Op<GteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = GteOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LtOp
	//===----------------------------------------------------------------------===//

	class LtOp;

	class LtOpAdaptor : public OpAdaptor<LtOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class LtOp : public mlir::Op<LtOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = LtOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LteOp
	//===----------------------------------------------------------------------===//

	class LteOp;

	class LteOpAdaptor : public OpAdaptor<LteOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class LteOp : public mlir::Op<LteOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = LteOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::NegateOp
	//===----------------------------------------------------------------------===//

	class NegateOp;

	class NegateOpAdaptor : public OpAdaptor<NegateOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class NegateOp : public mlir::Op<NegateOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait, DistributableInterface::Trait, NegateOpDistributionInterface::Trait, MulOpDistributionInterface::Trait, DivOpDistributionInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = NegateOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value value);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AddOp
	//===----------------------------------------------------------------------===//

	class AddOp;

	class AddOpAdaptor : public OpAdaptor<AddOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::OpTrait::IsCommutative, mlir::MemoryEffectOpInterface::Trait, NegateOpDistributionInterface::Trait, MulOpDistributionInterface::Trait, DivOpDistributionInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AddOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SubOp
	//===----------------------------------------------------------------------===//

	class SubOp;

	class SubOpAdaptor : public OpAdaptor<SubOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class SubOp : public mlir::Op<SubOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait, NegateOpDistributionInterface::Trait, MulOpDistributionInterface::Trait, DivOpDistributionInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SubOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MulOp
	//===----------------------------------------------------------------------===//

	class MulOp;

	class MulOpAdaptor : public OpAdaptor<MulOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class MulOp : public mlir::Op<MulOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait, DistributableInterface::Trait, NegateOpDistributionInterface::Trait, MulOpDistributionInterface::Trait, DivOpDistributionInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MulOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DivOp
	//===----------------------------------------------------------------------===//

	class DivOp;

	class DivOpAdaptor : public OpAdaptor<DivOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class DivOp : public mlir::Op<DivOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait,DistributableInterface::Trait, NegateOpDistributionInterface::Trait, MulOpDistributionInterface::Trait, DivOpDistributionInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = DivOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PowOp
	//===----------------------------------------------------------------------===//

	class PowOp;

	class PowOpAdaptor : public OpAdaptor<PowOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value base();
		mlir::Value exponent();
	};

	class PowOp : public mlir::Op<PowOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = PowOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value base, mlir::Value exponent);
		void print(mlir::OpAsmPrinter& printer);
		static void getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value base();
		mlir::Value exponent();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::NDimsOp
	//===----------------------------------------------------------------------===//

	class NDimsOp;

	class NDimsOpAdaptor : public OpAdaptor<NDimsOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
	};

	class NDimsOp : public mlir::Op<NDimsOp, mlir::OpTrait::AtLeastNOperands<1>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = NDimsOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value memory);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Type resultType();
		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SizeOp
	//===----------------------------------------------------------------------===//

	class SizeOp;

	class SizeOpAdaptor : public OpAdaptor<SizeOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
		mlir::Value index();
	};

	class SizeOp : public mlir::Op<SizeOp, mlir::OpTrait::AtLeastNOperands<1>::Impl, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = SizeOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value memory, mlir::Value index = nullptr);
		void print(mlir::OpAsmPrinter& printer);

		bool hasIndex();

		mlir::Type resultType();
		mlir::Value memory();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::FillOp
	//===----------------------------------------------------------------------===//

	class FillOp;

	class FillOpAdaptor : public OpAdaptor<FillOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value value();
		mlir::Value memory();
	};

	class FillOp : public mlir::Op<FillOp, mlir::OpTrait::NOperands<2>::Impl, mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = FillOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value, mlir::Value memory);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value value();
		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DerOp
	//===----------------------------------------------------------------------===//

	class DerOp;

	class DerOpAdaptor : public OpAdaptor<DerOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class DerOp : public mlir::Op<DerOp, mlir::OpTrait::OneOperand, mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = DerOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType, mlir::Value operand);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PrintOp
	//===----------------------------------------------------------------------===//

	class PrintOp;

	class PrintOpAdaptor : public OpAdaptor<PrintOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange values();
	};

	class PrintOp : public mlir::Op<PrintOp, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = PrintOpAdaptor;

		static llvm::StringRef getOperationName();
		static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::ValueRange values);
		void print(mlir::OpAsmPrinter& printer);
		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange values();
	};
}
