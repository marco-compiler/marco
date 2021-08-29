#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Type.h"

namespace marco::codegen::ida
{
	//===----------------------------------------------------------------------===//
	// Ida::ConstantValueOp
	//===----------------------------------------------------------------------===//

	class ConstantValueOp : public mlir::Op<ConstantValueOp,
																		mlir::OpTrait::ZeroRegion,
																		mlir::OpTrait::OneResult,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::ConstantLike>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.constant";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> operands);

		mlir::Attribute value();
		mlir::Type resultType();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AllocUserDataOp
	//===----------------------------------------------------------------------===//

	class AllocUserDataOp : public mlir::Op<AllocUserDataOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.alloc_user_data";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value neq, mlir::Value nnz);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value neq();
		mlir::Value nnz();
	};

	//===----------------------------------------------------------------------===//
	// Ida::FreeUserDataOp
	//===----------------------------------------------------------------------===//

	class FreeUserDataOp : public mlir::Op<FreeUserDataOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.free_user_data";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::SetInitialValueOp
	//===----------------------------------------------------------------------===//

	class SetInitialValueOp : public mlir::Op<SetInitialValueOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.set_initial_value";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value value, mlir::Value isState);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value value();
		mlir::Value isState();
	};

	//===----------------------------------------------------------------------===//
	// Ida::InitOp
	//===----------------------------------------------------------------------===//

	class InitOp : public mlir::Op<InitOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.init";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::StepOp
	//===----------------------------------------------------------------------===//

	class StepOp : public mlir::Op<StepOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.step";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddTimeOp
	//===----------------------------------------------------------------------===//

	class AddTimeOp : public mlir::Op<AddTimeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_time";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value stop);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value start();
		mlir::Value stop();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddToleranceOp
	//===----------------------------------------------------------------------===//

	class AddToleranceOp : public mlir::Op<AddToleranceOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_tolerance";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value relTol, mlir::Value absTol);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value relTol();
		mlir::Value absTol();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddRowLengthOp
	//===----------------------------------------------------------------------===//

	class AddRowLengthOp : public mlir::Op<AddRowLengthOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_row_length";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value rowLength);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value rowLength();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddDimensionOp
	//===----------------------------------------------------------------------===//

	class AddDimensionOp : public mlir::Op<AddDimensionOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_dimension";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value min, mlir::Value max);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value min();
		mlir::Value max();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddResidualOp
	//===----------------------------------------------------------------------===//

	class AddResidualOp : public mlir::Op<AddResidualOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_residual";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddJacobianOp
	//===----------------------------------------------------------------------===//

	class AddJacobianOp : public mlir::Op<AddJacobianOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_jacobian";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::GetTimeOp
	//===----------------------------------------------------------------------===//

	class GetTimeOp : public mlir::Op<GetTimeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.get_time";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::GetVariableOp
	//===----------------------------------------------------------------------===//

	class GetVariableOp : public mlir::Op<GetVariableOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.get_variable";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
	};
}
