#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>

#include "Attribute.h"
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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		OpaquePointerType resultType();
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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		BooleanType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::SetInitialValueOp
	//===----------------------------------------------------------------------===//

	class SetInitialValueOp : public mlir::Op<SetInitialValueOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<5>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.set_initial_value";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value length, mlir::Value value, mlir::Value isState);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value length();
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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		BooleanType resultType();
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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		BooleanType resultType();
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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		marco::codegen::modelica::RealType resultType();
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
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		marco::codegen::modelica::RealType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Ida::GetDerivativeOp
	//===----------------------------------------------------------------------===//

	class GetDerivativeOp : public mlir::Op<GetDerivativeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.get_derivative";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		marco::codegen::modelica::RealType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddNewLambdaAccessOp
	//===----------------------------------------------------------------------===//

	class AddNewLambdaAccessOp : public mlir::Op<AddNewLambdaAccessOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_new_lambda_access";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value indices);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value offset();
		mlir::Value indices();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddLambdaAccessOp
	//===----------------------------------------------------------------------===//

	class AddLambdaAccessOp : public mlir::Op<AddLambdaAccessOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_lambda_access";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value offset, mlir::Value indices);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value offset();
		mlir::Value indices();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddLambdaDimensionOp
	//===----------------------------------------------------------------------===//

	class AddLambdaDimensionOp : public mlir::Op<AddLambdaDimensionOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_lambda_dimension";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value dimension);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value dimension();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaConstantOp
	//===----------------------------------------------------------------------===//

	class LambdaConstantOp : public mlir::Op<LambdaConstantOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_constant";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value constant);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value constant();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaTimeOp
	//===----------------------------------------------------------------------===//

	class LambdaTimeOp : public mlir::Op<LambdaTimeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_time";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaScalarVariableOp
	//===----------------------------------------------------------------------===//

	class LambdaScalarVariableOp : public mlir::Op<LambdaScalarVariableOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_scalar_variable";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value offset();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaScalarDerivativeOp
	//===----------------------------------------------------------------------===//

	class LambdaScalarDerivativeOp : public mlir::Op<LambdaScalarDerivativeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_scalar_derivative";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value offset();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaVectorVariableOp
	//===----------------------------------------------------------------------===//

	class LambdaVectorVariableOp : public mlir::Op<LambdaVectorVariableOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_vector_variable";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value index);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value offset();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaVectorDerivativeOp
	//===----------------------------------------------------------------------===//

	class LambdaVectorDerivativeOp : public mlir::Op<LambdaVectorDerivativeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_vector_derivative";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value offset, mlir::Value index);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value offset();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAddOp
	//===----------------------------------------------------------------------===//

	class LambdaAddOp : public mlir::Op<LambdaAddOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_add";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaSubOp
	//===----------------------------------------------------------------------===//

	class LambdaSubOp : public mlir::Op<LambdaSubOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_sub";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaMulOp
	//===----------------------------------------------------------------------===//

	class LambdaMulOp : public mlir::Op<LambdaMulOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_mul";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaDivOp
	//===----------------------------------------------------------------------===//

	class LambdaDivOp : public mlir::Op<LambdaDivOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_div";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaPowOp
	//===----------------------------------------------------------------------===//

	class LambdaPowOp : public mlir::Op<LambdaPowOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_pow";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAtan2Op
	//===----------------------------------------------------------------------===//

	class LambdaAtan2Op : public mlir::Op<LambdaAtan2Op,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_atan2";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value leftIndex, mlir::Value rightIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value leftIndex();
		mlir::Value rightIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaNegateOp
	//===----------------------------------------------------------------------===//

	class LambdaNegateOp : public mlir::Op<LambdaNegateOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_negate";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAbsOp
	//===----------------------------------------------------------------------===//

	class LambdaAbsOp : public mlir::Op<LambdaAbsOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_abs";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaSignOp
	//===----------------------------------------------------------------------===//

	class LambdaSignOp : public mlir::Op<LambdaSignOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_sign";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaSqrtOp
	//===----------------------------------------------------------------------===//

	class LambdaSqrtOp : public mlir::Op<LambdaSqrtOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_sqrt";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaExpOp
	//===----------------------------------------------------------------------===//

	class LambdaExpOp : public mlir::Op<LambdaExpOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_exp";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaLogOp
	//===----------------------------------------------------------------------===//

	class LambdaLogOp : public mlir::Op<LambdaLogOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_log";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaLog10Op
	//===----------------------------------------------------------------------===//

	class LambdaLog10Op : public mlir::Op<LambdaLog10Op,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_log10";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaSinOp
	//===----------------------------------------------------------------------===//

	class LambdaSinOp : public mlir::Op<LambdaSinOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_sin";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaCosOp
	//===----------------------------------------------------------------------===//

	class LambdaCosOp : public mlir::Op<LambdaCosOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_cos";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaTanOp
	//===----------------------------------------------------------------------===//

	class LambdaTanOp : public mlir::Op<LambdaTanOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_tan";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAsinOp
	//===----------------------------------------------------------------------===//

	class LambdaAsinOp : public mlir::Op<LambdaAsinOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_asin";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAcosOp
	//===----------------------------------------------------------------------===//

	class LambdaAcosOp : public mlir::Op<LambdaAcosOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_acos";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaAtanOp
	//===----------------------------------------------------------------------===//

	class LambdaAtanOp : public mlir::Op<LambdaAtanOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_atan";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaSinhOp
	//===----------------------------------------------------------------------===//

	class LambdaSinhOp : public mlir::Op<LambdaSinhOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_sinh";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaCoshOp
	//===----------------------------------------------------------------------===//

	class LambdaCoshOp : public mlir::Op<LambdaCoshOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_cosh";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LambdaTanhOp
	//===----------------------------------------------------------------------===//

	class LambdaTanhOp : public mlir::Op<LambdaTanhOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.lambda_tanh";
		}

		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value operandIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value operandIndex();
	};
}
