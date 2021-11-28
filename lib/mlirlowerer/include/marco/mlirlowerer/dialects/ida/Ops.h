#pragma once

#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <marco/mlirlowerer/passes/model/Equation.h>
#include <marco/mlirlowerer/passes/model/Model.h>
#include <marco/mlirlowerer/passes/model/Variable.h>
#include <mlir/IR/FunctionSupport.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Attribute.h"
#include "Type.h"

namespace marco::codegen::ida
{
	using OffsetMap = std::map<model::Variable, int64_t>;

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

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> operands);

		mlir::Attribute value();
		mlir::Type resultType();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AllocDataOp
	//===----------------------------------------------------------------------===//

	class AllocDataOp : public mlir::Op<AllocDataOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.alloc_data";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value equationsNumber);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		OpaquePointerType resultType();
		mlir::ValueRange args();
		mlir::Value equationsNumber();
	};

	//===----------------------------------------------------------------------===//
	// Ida::InitOp
	//===----------------------------------------------------------------------===//

	class InitOp : public mlir::Op<InitOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.init";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value threads);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		BooleanType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value threads();
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

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
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
	// Ida::FreeDataOp
	//===----------------------------------------------------------------------===//

	class FreeDataOp : public mlir::Op<FreeDataOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.free_data";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
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
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_time";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value end, mlir::Value step);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value start();
		mlir::Value end();
		mlir::Value step();
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

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
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
	// Ida::AddColumnIndexOp
	//===----------------------------------------------------------------------===//

	class AddColumnIndexOp : public mlir::Op<AddColumnIndexOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_column_index";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value rowIndex, mlir::Value accessIndex);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value rowIndex();
		mlir::Value accessIndex();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddEqDimensionOp
	//===----------------------------------------------------------------------===//

	class AddEqDimensionOp : public mlir::Op<AddEqDimensionOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_eq_dimension";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value start, mlir::Value end);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value start();
		mlir::Value end();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddResidualOp
	//===----------------------------------------------------------------------===//

	class AddResidualOp : public mlir::Op<AddResidualOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_residual";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value residualAddress);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value residualAddress();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddJacobianOp
	//===----------------------------------------------------------------------===//

	class AddJacobianOp : public mlir::Op<AddJacobianOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_jacobian";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value jacobianAddress);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value jacobianAddress();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddVariableOp
	//===----------------------------------------------------------------------===//

	class AddVariableOp : public mlir::Op<AddVariableOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_variable";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value array, mlir::Value isState);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value array();
		mlir::Value isState();
	};

	//===----------------------------------------------------------------------===//
	// Ida::AddVarAccessOp
	//===----------------------------------------------------------------------===//

	class AddVarAccessOp : public mlir::Op<AddVarAccessOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<4>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.add_var_access";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value variable, mlir::Value offsets, mlir::Value inductions);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		IntegerType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value variable();
		mlir::Value offsets();
		mlir::Value inductions();
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

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		modelica::RealType resultType();
		mlir::ValueRange args();
		mlir::Value userData();
	};

	//===----------------------------------------------------------------------===//
	// Ida::UpdateVariableOp
	//===----------------------------------------------------------------------===//

	class UpdateVariableOp : public mlir::Op<UpdateVariableOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.update_variable";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value array);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value array();
	};

	//===----------------------------------------------------------------------===//
	// Ida::UpdateDerivativeOp
	//===----------------------------------------------------------------------===//

	class UpdateDerivativeOp : public mlir::Op<UpdateDerivativeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<3>::Impl,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.update_derivative";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData, mlir::Value index, mlir::Value array);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange args();
		mlir::Value userData();
		mlir::Value index();
		mlir::Value array();
	};

	//===----------------------------------------------------------------------===//
	// Ida::ResidualFunctionOp
	//===----------------------------------------------------------------------===//

	class ResidualFunctionOp : public mlir::Op<ResidualFunctionOp,
																		mlir::OpTrait::OneRegion,
																		mlir::OpTrait::ZeroResult,
																		mlir::OpTrait::ZeroSuccessor,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::NoTerminator,
																		mlir::OpTrait::IsIsolatedFromAbove,
																		mlir::OpTrait::FunctionLike,
																		mlir::CallableOpInterface::Trait,
																		mlir::SymbolOpInterface::Trait,
																		mlir::OpTrait::AutomaticAllocationScope>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.residual_function";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation, OffsetMap offsetMap);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Region* getCallableRegion();
		llvm::ArrayRef<mlir::Type> getCallableResults();

		llvm::StringRef name();

		private:
		friend class mlir::OpTrait::FunctionLike<ResidualFunctionOp>;

		unsigned int getNumFuncArguments();
		unsigned int getNumFuncResults();
	};

	//===----------------------------------------------------------------------===//
	// Ida::JacobianFunctionOp
	//===----------------------------------------------------------------------===//

	class JacobianFunctionOp : public mlir::Op<JacobianFunctionOp,
																		mlir::OpTrait::OneRegion,
																		mlir::OpTrait::ZeroResult,
																		mlir::OpTrait::ZeroSuccessor,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::IsIsolatedFromAbove,
																		mlir::OpTrait::FunctionLike,
																		mlir::CallableOpInterface::Trait,
																		mlir::SymbolOpInterface::Trait,
																		mlir::OpTrait::AutomaticAllocationScope>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.jacobian_function";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, model::Model& model, model::Equation& equation, OffsetMap offsetMap);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Region* getCallableRegion();
		llvm::ArrayRef<mlir::Type> getCallableResults();

		llvm::StringRef name();

		private:
		friend class mlir::OpTrait::FunctionLike<JacobianFunctionOp>;

		unsigned int getNumFuncArguments();
		unsigned int getNumFuncResults();
	};

	//===----------------------------------------------------------------------===//
	// Ida::FunctionTerminatorOp
	//===----------------------------------------------------------------------===//

	class FunctionTerminatorOp : public mlir::Op<FunctionTerminatorOp,
																		mlir::OpTrait::ZeroRegion,
																		mlir::OpTrait::ZeroResult,
																		mlir::OpTrait::ZeroSuccessor,
																		mlir::OpTrait::OneOperand,
																		mlir::OpTrait::HasParent<ResidualFunctionOp, JacobianFunctionOp>::Impl,
																		mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.function_terminator";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value returnValue);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::ValueRange args();
		mlir::Value returnValue();
	};

	//===----------------------------------------------------------------------===//
	// Ida::FuncAddressOfOp
	//===----------------------------------------------------------------------===//

	class FuncAddressOfOp : public mlir::Op<FuncAddressOfOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::ZeroOperands,
																mlir::OpTrait::OneResult> 
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.func_addressof";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::Type type);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::StringRef callee();
	};

	//===----------------------------------------------------------------------===//
	// Ida::LoadPointerOp
	//===----------------------------------------------------------------------===//

	class LoadPointerOp : public mlir::Op<LoadPointerOp,
																		mlir::OpTrait::ZeroRegion,
																		mlir::OpTrait::NOperands<2>::Impl,
																		mlir::OpTrait::OneResult,
																		mlir::MemoryEffectOpInterface::Trait,
																		mlir::ViewLikeOpInterface::Trait>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.load_ptr";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value pointer, mlir::Value offset);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);
		mlir::Value getViewSource();

		mlir::Type resultType();
		mlir::ValueRange args();
		mlir::Value pointer();
		mlir::Value offset();
	};

	//===----------------------------------------------------------------------===//
	// Ida::PrintStatisticsOp
	//===----------------------------------------------------------------------===//

	class PrintStatisticsOp : public mlir::Op<PrintStatisticsOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.print_stats";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value userData);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::ValueRange args();
		mlir::Value userData();
	};
}
