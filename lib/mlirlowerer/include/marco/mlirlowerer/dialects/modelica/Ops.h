#pragma once

#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/IR/FunctionSupport.h>
#include <mlir/IR/OpDefinition.h>

#include "Attribute.h"
#include "Traits.h"
#include "Type.h"

namespace marco::codegen::modelica
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

	class PackOp : public mlir::Op<PackOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::VariadicOperands,
																mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = PackOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.pack";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange values);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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
		unsigned int index();
	};

	class ExtractOp : public mlir::Op<ExtractOp,
																	 mlir::OpTrait::ZeroRegion,
																	 mlir::OpTrait::OneOperand,
																	 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = ExtractOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.extract";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value packedValue, unsigned int index);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class SimulationOp : public mlir::Op<SimulationOp,
																			mlir::OpTrait::NRegions<2>::Impl,
																			mlir::OpTrait::ZeroOperands,
																			mlir::OpTrait::ZeroResult,
																			mlir::RegionBranchOpInterface::Trait,
																			mlir::OpTrait::IsIsolatedFromAbove>
	{
		public:
		using Op::Op;
		using Adaptor = SimulationOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.simulation";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr variableNames, RealAttribute startTime, RealAttribute endTime, RealAttribute timeStep, RealAttribute relTol, RealAttribute absTol, mlir::TypeRange vars);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		RealAttribute startTime();
		RealAttribute endTime();
		RealAttribute timeStep();
		RealAttribute relTol();
		RealAttribute absTol();

		mlir::Region& init();
		mlir::Region& body();

		mlir::Value getVariableAllocation(mlir::Value var);

		mlir::Value time();

        mlir::ArrayAttr variableNames();
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

	class EquationOp : public mlir::Op<EquationOp,
																		mlir::OpTrait::OneRegion,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::ZeroResult,
																		EquationInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = EquationOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.equation";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class ForEquationOp : public mlir::Op<ForEquationOp,
																			 mlir::OpTrait::NRegions<2>::Impl,
																			 mlir::OpTrait::ZeroOperands,
																			 mlir::OpTrait::ZeroResult,
																			 EquationInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ForEquationOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.for_equation";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, size_t inductionsAmount);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class InductionOpAdaptor : public OpAdaptor<InductionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		long start();
		long end();
	};

	class InductionOp : public mlir::Op<InductionOp,
																		 mlir::OpTrait::ZeroRegion,
																		 mlir::OpTrait::ZeroOperands,
																		 mlir::OpTrait::OneResult,
																		 mlir::OpTrait::HasParent<ForEquationOp>::Impl>
	{
		public:
		using Op::Op;
		using Adaptor = InductionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.induction";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, long start, long end);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class EquationSidesOp : public mlir::Op<EquationSidesOp,
																				 mlir::OpTrait::ZeroRegion,
																				 mlir::OpTrait::VariadicOperands,
																				 mlir::OpTrait::ZeroResult,
																				 mlir::OpTrait::HasParent<EquationOp, ForEquationOp>::Impl,
																				 mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = EquationSidesOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.equation_sides";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange lhs, mlir::ValueRange rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange lhs();
		mlir::ValueRange rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::FunctionOp
	//===----------------------------------------------------------------------===//

	class FunctionOp;

	class FunctionOpAdaptor : public OpAdaptor<FunctionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		llvm::ArrayRef<mlir::Attribute> argsNames();
		llvm::ArrayRef<mlir::Attribute> resultsNames();
	};

	class FunctionOp : public mlir::Op<FunctionOp,
																		mlir::OpTrait::OneRegion,
																		mlir::OpTrait::ZeroResult,
																		mlir::OpTrait::ZeroSuccessor,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::NoTerminator,
																		mlir::OpTrait::IsIsolatedFromAbove,
																		mlir::OpTrait::FunctionLike,
																		mlir::CallableOpInterface::Trait,
																		mlir::SymbolOpInterface::Trait,
																		mlir::OpTrait::AutomaticAllocationScope,
																		ClassInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = FunctionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.function";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::FunctionType type, llvm::ArrayRef<llvm::StringRef> argsNames, llvm::ArrayRef<llvm::StringRef> resultsNames);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::FunctionType type, mlir::ArrayAttr argsNames, mlir::ArrayAttr resultsNames);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Region* getCallableRegion();
		llvm::ArrayRef<mlir::Type> getCallableResults();

		llvm::StringRef name();

		llvm::ArrayRef<mlir::Attribute> argsNames();
		llvm::ArrayRef<mlir::Attribute> resultsNames();

		bool hasDerivative();

		void getMembers(llvm::SmallVectorImpl<mlir::Value>& members, llvm::SmallVectorImpl<llvm::StringRef>& names);

		private:
		friend class mlir::OpTrait::FunctionLike<FunctionOp>;

		unsigned int getNumFuncArguments();
		unsigned int getNumFuncResults();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::FunctionTerminatorOp
	//===----------------------------------------------------------------------===//

	class FunctionTerminatorOp;

	class FunctionTerminatorOpAdaptor : public OpAdaptor<FunctionTerminatorOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class FunctionTerminatorOp : public mlir::Op<FunctionTerminatorOp,
																							mlir::OpTrait::ZeroRegion,
																							mlir::OpTrait::ZeroResult,
																							mlir::OpTrait::ZeroSuccessor,
																							mlir::OpTrait::ZeroOperands,
																							mlir::OpTrait::HasParent<FunctionOp>::Impl,
																							mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = FunctionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.function_terminator";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DerFunctionOp
	//===----------------------------------------------------------------------===//

	class DerFunctionOp;

	class DerFunctionOpAdaptor : public OpAdaptor<DerFunctionOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

	};

	class DerFunctionOp : public mlir::Op<DerFunctionOp,
																			 mlir::OpTrait::ZeroRegion,
																			 mlir::OpTrait::ZeroResult,
																			 mlir::OpTrait::ZeroSuccessor,
																			 mlir::OpTrait::ZeroOperands,
																			 mlir::OpTrait::IsIsolatedFromAbove,
																			 mlir::CallableOpInterface::Trait,
																			 mlir::SymbolOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = DerFunctionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.der_function";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, llvm::StringRef derivedFunction, llvm::ArrayRef<llvm::StringRef> independentVariables);
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, llvm::StringRef derivedFunction, llvm::ArrayRef<mlir::Attribute> independentVariables);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Region* getCallableRegion();
		llvm::ArrayRef<mlir::Type> getCallableResults();

		llvm::StringRef name();
		llvm::StringRef derivedFunction();
		llvm::ArrayRef<mlir::Attribute> independentVariables();
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

	class ConstantOp : public mlir::Op<ConstantOp,
																		mlir::OpTrait::ZeroRegion,
																		mlir::OpTrait::OneResult,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::ConstantLike,
																		DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ConstantOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.constant";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Attribute value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> operands);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

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

	class CastOp : public mlir::Op<CastOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = CastOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.cast";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& Builder, mlir::OperationState& state, mlir::Value value, mlir::Type resultType);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		mlir::OpFoldResult fold(mlir::ArrayRef<mlir::Attribute> operands);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Value value();
		mlir::Type resultType();
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

	class AssignmentOp : public mlir::Op<AssignmentOp,
																			mlir::OpTrait::ZeroRegion,
																			mlir::OpTrait::ZeroResult,
																			mlir::OpTrait::VariadicOperands,
																			mlir::MemoryEffectOpInterface::Trait,
																			DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AssignmentOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.assignment";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::Value destination);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

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

	class CallOp : public mlir::Op<CallOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::VariadicResults,
																mlir::OpTrait::VariadicOperands,
																mlir::MemoryEffectOpInterface::Trait,
																mlir::CallOpInterface::Trait,
																VectorizableOpInterface::Trait,
																InvertibleOpInterface::Trait,
																DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = CallOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.call";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::StringRef callee, mlir::TypeRange results, mlir::ValueRange args, unsigned int movedResults = 0);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::CallInterfaceCallable getCallableForCallee();
		mlir::Operation::operand_range getArgOperands();

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::StringRef callee();
		mlir::TypeRange resultTypes();
		mlir::ValueRange args();
		unsigned int movedResults();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MemberCreateOp
	//===----------------------------------------------------------------------===//

	class MemberCreateOp;

	class MemberCreateOpAdaptor : public OpAdaptor<MemberCreateOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange dynamicDimensions();
	};

	class MemberCreateOp : public mlir::Op<MemberCreateOp,
																				mlir::OpTrait::ZeroRegion,
																				mlir::OpTrait::VariadicOperands,
																				mlir::OpTrait::OneResult,
																				mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MemberCreateOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.member_create";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, llvm::StringRef name, mlir::Type type, mlir::ValueRange dynamicDimensions, mlir::NamedAttrList attributes = {});
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		llvm::StringRef name();
		mlir::Type resultType();
		mlir::ValueRange dynamicDimensions();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MemberReadOp
	//===----------------------------------------------------------------------===//

	class MemberLoadOp;

	class MemberLoadOpAdaptor : public OpAdaptor<MemberLoadOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value member();
	};

	class MemberLoadOp : public mlir::Op<MemberLoadOp,
																			mlir::OpTrait::ZeroRegion,
																			mlir::OpTrait::OneOperand,
																			mlir::OpTrait::OneResult,
																			mlir::MemoryEffectOpInterface::Trait,
																			DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MemberLoadOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.member_load";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value member);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Type resultType();
		mlir::Value member();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MemberStoreOp
	//===----------------------------------------------------------------------===//

	class MemberStoreOp;

	class MemberStoreOpAdaptor : public OpAdaptor<MemberStoreOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value member();
		mlir::Value value();
	};

	class MemberStoreOp :public mlir::Op<MemberStoreOp,
																			 mlir::OpTrait::ZeroRegion,
																			 mlir::OpTrait::NOperands<2>::Impl,
																			 mlir::OpTrait::ZeroResult,
																			 mlir::MemoryEffectOpInterface::Trait,
																			 DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MemberStoreOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.member_store";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value member, mlir::Value value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Value member();
		mlir::Value value();
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

	class AllocaOp : public mlir::Op<AllocaOp,
																	mlir::OpTrait::ZeroRegion,
																	mlir::OpTrait::VariadicOperands,
																	mlir::OpTrait::OneResult,
																	mlir::MemoryEffectOpInterface::Trait,
																	DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AllocaOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.alloca";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape = {}, mlir::ValueRange dimensions = {}, bool constant = false);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		ArrayType resultType();
		mlir::ValueRange dynamicDimensions();
		bool isConstant();

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);
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
		bool isConstant();
	};

	class AllocOp : public mlir::Op<AllocOp,
																 mlir::OpTrait::ZeroRegion,
																 mlir::OpTrait::VariadicOperands,
																 mlir::OpTrait::OneResult,
																 mlir::MemoryEffectOpInterface::Trait,
																 HeapAllocator::Trait,
																 DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AllocOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.alloc";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type elementType, llvm::ArrayRef<long> shape = llvm::None, mlir::ValueRange dimensions = llvm::None, bool shouldBeFreed = true, bool isConstant = false);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		ArrayType resultType();
		mlir::ValueRange dynamicDimensions();
		bool isConstant();

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);
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

	class FreeOp : public mlir::Op<FreeOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::ZeroResult,
																mlir::MemoryEffectOpInterface::Trait,
																DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = FreeOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.free";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Value memory();

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ArrayCastOp
	//===----------------------------------------------------------------------===//

	/**
	 * This operation should be used only for two purposes: the first one is
	 * for function calls, to remove the allocation scope before passing the
	 * array as arguments to the functions, or to generalize the sizes to
	 * unknown ones; the second one is to cast from an array with unknown
	 * allocation scope to an array with a known one.
	 * The operation is NOT intended to be used to change the allocation scope
	 * between known ones (i.e. stack -> heap or heap -> stack), to cast the
	 * element type to a different one or to to specialize the shape to a fixed
	 * one.
	 */
	class ArrayCastOp;

	class ArrayCastOpAdaptor : public OpAdaptor<ArrayCastOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value memory();
	};

	class ArrayCastOp : public mlir::Op<ArrayCastOp,
																		 mlir::OpTrait::ZeroRegion,
																		 mlir::OpTrait::OneOperand,
																		 mlir::OpTrait::OneResult,
																		 mlir::ViewLikeOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ArrayCastOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.array_cast";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value, mlir::Type resultType);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Value getViewSource();

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

	class DimOp : public mlir::Op<DimOp,
															 mlir::OpTrait::ZeroRegion,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = DimOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.dim";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::Value dimension);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();
		mlir::OpFoldResult fold(mlir::ArrayRef<mlir::Attribute> operands);

		ArrayType getArrayType();
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

	class SubscriptionOp : public mlir::Op<SubscriptionOp,
																				mlir::OpTrait::ZeroRegion,
																				mlir::OpTrait::AtLeastNOperands<2>::Impl,
																				mlir::OpTrait::OneResult,
																				mlir::ViewLikeOpInterface::Trait,
																				DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SubscriptionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.subscription";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, mlir::ValueRange indexes);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value getViewSource();

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		ArrayType resultType();
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

	class LoadOp : public mlir::Op<LoadOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::AtLeastNOperands<1>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait,
																DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = LoadOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.load";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value memory, mlir::ValueRange indexes = llvm::None);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		ArrayType getArrayType();
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

	class StoreOp :public mlir::Op<StoreOp,
																 mlir::OpTrait::ZeroRegion,
																 mlir::OpTrait::AtLeastNOperands<2>::Impl,
																 mlir::OpTrait::ZeroResult,
																 mlir::MemoryEffectOpInterface::Trait,
																 DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = StoreOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.store";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory, mlir::ValueRange indexes = llvm::None);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		ArrayType getArrayType();
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
		bool canSourceBeForwarded();
	};

	class ArrayCloneOp :public mlir::Op<ArrayCloneOp,
																			mlir::OpTrait::ZeroRegion,
																			mlir::OpTrait::OneOperand,
																			mlir::OpTrait::OneResult,
																			mlir::MemoryEffectOpInterface::Trait,
																			HeapAllocator::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ArrayCloneOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.array_clone";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value source, ArrayType resultType, bool shouldBeFreed = true, bool canSourceBeForwarded = false);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		ArrayType resultType();
		mlir::Value source();
		bool canSourceBeForwarded();
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

	class IfOp : public mlir::Op<IfOp,
															mlir::OpTrait::NRegions<2>::Impl,
															mlir::OpTrait::ZeroResult,
															mlir::OpTrait::ZeroSuccessor,
															mlir::OpTrait::OneOperand,
															mlir::OpTrait::NoTerminator,
															mlir::RegionBranchOpInterface::Trait,
															DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = IfOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.if";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value cond, bool withElseRegion = false);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(::mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

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

	class ForOp : public mlir::Op<ForOp,
															 mlir::OpTrait::NRegions<3>::Impl,
															 mlir::OpTrait::VariadicOperands,
															 mlir::OpTrait::ZeroResult,
															 mlir::RegionBranchOpInterface::Trait,
															 DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ForOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.for";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args = llvm::None);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Region& condition();
		mlir::Region& body();
		mlir::Region& step();

		mlir::ValueRange args();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::WhileOp
	//===----------------------------------------------------------------------===//

	class WhileOp;

	class WhileOpAdaptor : public OpAdaptor<WhileOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class WhileOp : public mlir::Op<WhileOp,
																 mlir::OpTrait::NRegions<2>::Impl,
																 mlir::OpTrait::ZeroOperands,
																 mlir::OpTrait::ZeroResult,
																 mlir::LoopLikeOpInterface::Trait,
																 mlir::RegionBranchOpInterface::Trait,
																 DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = WhileOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.while";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		bool isDefinedOutsideOfLoop(mlir::Value value);
		mlir::Region& getLoopBody();
		mlir::LogicalResult moveOutOfLoop(llvm::ArrayRef<mlir::Operation*> ops);

		void getSuccessorRegions(llvm::Optional<unsigned> index, llvm::ArrayRef<mlir::Attribute> operands, llvm::SmallVectorImpl<mlir::RegionSuccessor>& regions);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Region& condition();
		mlir::Region& body();
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

	class ConditionOp : public mlir::Op<ConditionOp,
																		 mlir::OpTrait::ZeroRegion,
																		 mlir::OpTrait::VariadicOperands,
																		 mlir::OpTrait::ZeroResult,
																		 mlir::OpTrait::ZeroSuccessor,
																		 mlir::OpTrait::HasParent<ForOp, WhileOp>::Impl,
																		 mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = ConditionOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.condition";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args = llvm::None);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(::mlir::OpAsmPrinter& printer);
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

		mlir::ValueRange values();
	};

	class YieldOp : public mlir::Op<YieldOp,
																 mlir::OpTrait::ZeroRegion,
																 mlir::OpTrait::VariadicOperands,
																 mlir::OpTrait::ZeroResult,
																 mlir::OpTrait::HasParent<ForEquationOp, IfOp, ForOp, WhileOp, SimulationOp>::Impl,
																 mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = YieldOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.yield";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args = llvm::None);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange values();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::BreakOp
	//===----------------------------------------------------------------------===//

	class BreakOp;

	class BreakOpAdaptor : public OpAdaptor<BreakOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class BreakOp : public mlir::Op<BreakOp,
																 mlir::OpTrait::ZeroRegion,
																 mlir::OpTrait::ZeroOperands,
																 mlir::OpTrait::ZeroResult,
																 mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = BreakOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.break";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ReturnOp
	//===----------------------------------------------------------------------===//

	class ReturnOp;

	class ReturnOpAdaptor : public OpAdaptor<ReturnOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class ReturnOp : public mlir::Op<ReturnOp,
			mlir::OpTrait::ZeroRegion,
			mlir::OpTrait::ZeroOperands,
			mlir::OpTrait::ZeroResult,
			mlir::OpTrait::IsTerminator>
	{
		public:
		using Op::Op;
		using Adaptor = ReturnOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.return";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
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

	class NotOp : public mlir::Op<NotOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = NotOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.not";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class AndOp : public mlir::Op<AndOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::IsCommutative,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AndOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.and";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class OrOp : public mlir::Op<OrOp,
															mlir::OpTrait::NOperands<2>::Impl,
															mlir::OpTrait::IsCommutative,
															mlir::OpTrait::OneResult,
															mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = OrOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.or";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class EqOp : public mlir::Op<EqOp,
															mlir::OpTrait::NOperands<2>::Impl,
															mlir::OpTrait::IsCommutative,
															mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = EqOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.eq";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class NotEqOp : public mlir::Op<NotEqOp,
																 mlir::OpTrait::NOperands<2>::Impl,
																 mlir::OpTrait::IsCommutative,
																 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = NotEqOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.neq";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class GtOp : public mlir::Op<GtOp,
															mlir::OpTrait::NOperands<2>::Impl,
															mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = GtOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.gt";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class GteOp : public mlir::Op<GteOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = GteOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.gte";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class LtOp : public mlir::Op<LtOp,
															mlir::OpTrait::NOperands<2>::Impl,
															mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = LtOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.lt";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class LteOp : public mlir::Op<LteOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = LteOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.lte";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class NegateOp : public mlir::Op<NegateOp,
																	mlir::OpTrait::OneOperand,
																	mlir::OpTrait::OneResult,
																	mlir::MemoryEffectOpInterface::Trait,
																	InvertibleOpInterface::Trait,
																	DistributableInterface::Trait,
																	NegateOpDistributionInterface::Trait,
																	MulOpDistributionInterface::Trait,
																	DivOpDistributionInterface::Trait,
																	DerivativeInterface::Trait,
																	FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = NegateOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.neg";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

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

	class AddOp : public mlir::Op<AddOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult,
															 mlir::OpTrait::IsCommutative,
															 mlir::MemoryEffectOpInterface::Trait,
															 InvertibleOpInterface::Trait,
															 NegateOpDistributionInterface::Trait,
															 MulOpDistributionInterface::Trait,
															 DivOpDistributionInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AddOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.add";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AddElementWiseOp
	//===----------------------------------------------------------------------===//

	class AddElementWiseOp;

	class AddElementWiseOpAdaptor : public OpAdaptor<AddElementWiseOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class AddElementWiseOp : public mlir::Op<AddElementWiseOp,
																					mlir::OpTrait::NOperands<2>::Impl,
																					mlir::OpTrait::OneResult,
																					mlir::OpTrait::IsCommutative,
																					mlir::MemoryEffectOpInterface::Trait,
																					InvertibleOpInterface::Trait,
																					NegateOpDistributionInterface::Trait,
																					MulOpDistributionInterface::Trait,
																					DivOpDistributionInterface::Trait,
																					DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AddElementWiseOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.add_ew";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

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

	class SubOp : public mlir::Op<SubOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait,
															 InvertibleOpInterface::Trait,
															 NegateOpDistributionInterface::Trait,
															 MulOpDistributionInterface::Trait,
															 DivOpDistributionInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SubOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sub";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SubElementWiseOp
	//===----------------------------------------------------------------------===//

	class SubElementWiseOp;

	class SubElementWiseOpAdaptor : public OpAdaptor<SubElementWiseOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class SubElementWiseOp : public mlir::Op<SubElementWiseOp,
																					mlir::OpTrait::NOperands<2>::Impl,
																					mlir::OpTrait::OneResult,
																					mlir::MemoryEffectOpInterface::Trait,
																					InvertibleOpInterface::Trait,
																					NegateOpDistributionInterface::Trait,
																					MulOpDistributionInterface::Trait,
																					DivOpDistributionInterface::Trait,
																					DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SubElementWiseOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sub_ew";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

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

	class MulOp : public mlir::Op<MulOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait,
															 InvertibleOpInterface::Trait,
															 DistributableInterface::Trait,
															 NegateOpDistributionInterface::Trait,
															 MulOpDistributionInterface::Trait,
															 DivOpDistributionInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MulOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.mul";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MulElementWiseOp
	//===----------------------------------------------------------------------===//

	class MulElementWiseOp;

	class MulElementWiseOpAdaptor : public OpAdaptor<MulElementWiseOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class MulElementWiseOp : public mlir::Op<MulElementWiseOp,
																					mlir::OpTrait::NOperands<2>::Impl,
																					mlir::OpTrait::OneResult,
																					mlir::MemoryEffectOpInterface::Trait,
																					InvertibleOpInterface::Trait,
																					DistributableInterface::Trait,
																					NegateOpDistributionInterface::Trait,
																					MulOpDistributionInterface::Trait,
																					DivOpDistributionInterface::Trait,
																					DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = MulElementWiseOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.mul_ew";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

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

	class DivOp : public mlir::Op<DivOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait,
															 InvertibleOpInterface::Trait,
															 DistributableInterface::Trait,
															 NegateOpDistributionInterface::Trait,
															 MulOpDistributionInterface::Trait,
															 DivOpDistributionInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = DivOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.div";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Type resultType();
		mlir::Value lhs();
		mlir::Value rhs();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DivElementWiseOp
	//===----------------------------------------------------------------------===//

	class DivElementWiseOp;

	class DivElementWiseOpAdaptor : public OpAdaptor<DivElementWiseOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value lhs();
		mlir::Value rhs();
	};

	class DivElementWiseOp : public mlir::Op<DivElementWiseOp,
																					mlir::OpTrait::NOperands<2>::Impl,
																					mlir::OpTrait::OneResult,
																					mlir::MemoryEffectOpInterface::Trait,
																					InvertibleOpInterface::Trait,
																					DistributableInterface::Trait,
																					NegateOpDistributionInterface::Trait,
																					MulOpDistributionInterface::Trait,
																					DivOpDistributionInterface::Trait,
																					DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = DivElementWiseOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.div_ew";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value lhs, mlir::Value rhs);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::LogicalResult invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult);

		mlir::Value distribute(mlir::OpBuilder& builder);
		mlir::Value distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType);
		mlir::Value distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);
		mlir::Value distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

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

	class PowOp : public mlir::Op<PowOp,
															 mlir::OpTrait::NOperands<2>::Impl,
															 mlir::OpTrait::OneResult,
															 mlir::MemoryEffectOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = PowOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.pow";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		static void getCanonicalizationPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value base();
		mlir::Value exponent();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PowElementWiseOp
	//===----------------------------------------------------------------------===//

	class PowElementWiseOp;

	class PowElementWiseOpAdaptor : public OpAdaptor<PowElementWiseOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value base();
		mlir::Value exponent();
	};

	class PowElementWiseOp : public mlir::Op<PowElementWiseOp,
																					mlir::OpTrait::NOperands<2>::Impl,
																					mlir::OpTrait::OneResult,
																					mlir::MemoryEffectOpInterface::Trait,
																					DerivativeInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = PowElementWiseOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.pow_ew";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value base, mlir::Value exponent);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		mlir::Type resultType();
		mlir::Value base();
		mlir::Value exponent();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AbsOp
	//===----------------------------------------------------------------------===//

	class AbsOp;

	class AbsOpAdaptor : public OpAdaptor<AbsOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class AbsOp : public mlir::Op<AbsOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AbsOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.abs";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SignOp
	//===----------------------------------------------------------------------===//

	class SignOp;

	class SignOpAdaptor : public OpAdaptor<SignOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class SignOp : public mlir::Op<SignOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SignOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sign";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SqrtOp
	//===----------------------------------------------------------------------===//

	class SqrtOp;

	class SqrtOpAdaptor : public OpAdaptor<SqrtOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class SqrtOp : public mlir::Op<SqrtOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SqrtOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sqrt";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SinOp
	//===----------------------------------------------------------------------===//

	class SinOp;

	class SinOpAdaptor : public OpAdaptor<SinOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class SinOp : public mlir::Op<SinOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SinOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sin";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CosOp
	//===----------------------------------------------------------------------===//

	class CosOp;

	class CosOpAdaptor : public OpAdaptor<CosOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class CosOp : public mlir::Op<CosOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = CosOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.cos";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::TanOp
	//===----------------------------------------------------------------------===//

	class TanOp;

	class TanOpAdaptor : public OpAdaptor<TanOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class TanOp : public mlir::Op<TanOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = TanOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.tan";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AsinOp
	//===----------------------------------------------------------------------===//

	class AsinOp;

	class AsinOpAdaptor : public OpAdaptor<AsinOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class AsinOp : public mlir::Op<AsinOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AsinOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.asin";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AcosOp
	//===----------------------------------------------------------------------===//

	class AcosOp;

	class AcosOpAdaptor : public OpAdaptor<AcosOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class AcosOp : public mlir::Op<AcosOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AcosOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.acos";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::AtanOp
	//===----------------------------------------------------------------------===//

	class AtanOp;

	class AtanOpAdaptor : public OpAdaptor<AtanOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class AtanOp : public mlir::Op<AtanOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AtanOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.atan";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::Atan2Op
	//===----------------------------------------------------------------------===//

	class Atan2Op;

	class Atan2OpAdaptor : public OpAdaptor<Atan2Op>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value y();
		mlir::Value x();
	};

	class Atan2Op : public mlir::Op<Atan2Op,
																 mlir::OpTrait::NOperands<2>::Impl,
																 mlir::OpTrait::OneResult,
																 VectorizableOpInterface::Trait,
																 DerivativeInterface::Trait,
																 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = Atan2OpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.atan2";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value y, mlir::Value x);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value y();
		mlir::Value x();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SinhOp
	//===----------------------------------------------------------------------===//

	class SinhOp;

	class SinhOpAdaptor : public OpAdaptor<SinhOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class SinhOp : public mlir::Op<SinhOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SinhOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sinh";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::CoshOp
	//===----------------------------------------------------------------------===//

	class CoshOp;

	class CoshOpAdaptor : public OpAdaptor<CoshOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class CoshOp : public mlir::Op<CoshOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = CoshOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.cosh";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::TanhOp
	//===----------------------------------------------------------------------===//

	class TanhOp;

	class TanhOpAdaptor : public OpAdaptor<TanhOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class TanhOp : public mlir::Op<TanhOp,
																mlir::OpTrait::OneOperand,
																mlir::OpTrait::OneResult,
																VectorizableOpInterface::Trait,
																DerivativeInterface::Trait,
																FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = TanhOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.tanh";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ExpOp
	//===----------------------------------------------------------------------===//

	class ExpOp;

	class ExpOpAdaptor : public OpAdaptor<ExpOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value exponent();
	};

	class ExpOp : public mlir::Op<ExpOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ExpOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.exp";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value exponent);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value exponent();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LogOp
	//===----------------------------------------------------------------------===//

	class LogOp;

	class LogOpAdaptor : public OpAdaptor<LogOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class LogOp : public mlir::Op<LogOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult,
															 VectorizableOpInterface::Trait,
															 DerivativeInterface::Trait,
															 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = LogOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.log";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::Log10Op
	//===----------------------------------------------------------------------===//

	class Log10Op;

	class Log10OpAdaptor : public OpAdaptor<Log10Op>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value operand();
	};

	class Log10Op : public mlir::Op<Log10Op,
																 mlir::OpTrait::OneOperand,
																 mlir::OpTrait::OneResult,
																 VectorizableOpInterface::Trait,
																 DerivativeInterface::Trait,
																 FoldableOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = Log10OpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.log10";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::ValueRange getArgs();
		unsigned int getArgExpectedRank(unsigned int argIndex);
		mlir::ValueRange scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes);

		mlir::ValueRange derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives);
		void getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived);
		void getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions);

		void foldConstants(mlir::OpBuilder& builder);

		mlir::Type resultType();
		mlir::Value operand();
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

	class NDimsOp : public mlir::Op<NDimsOp,
																 mlir::OpTrait::AtLeastNOperands<1>::Impl,
																 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = NDimsOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.ndims";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
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

	class SizeOp : public mlir::Op<SizeOp,
																mlir::OpTrait::AtLeastNOperands<1>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SizeOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.size";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value memory, mlir::Value index = nullptr);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		bool hasIndex();

		mlir::Type resultType();
		mlir::Value memory();
		mlir::Value index();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::IdentityOp
	//===----------------------------------------------------------------------===//

	class IdentityOp;

	class IdentityOpAdaptor : public OpAdaptor<IdentityOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value size();
	};

	class IdentityOp : public mlir::Op<IdentityOp,
																		mlir::OpTrait::OneOperand,
																		mlir::OpTrait::OneResult,
																		mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = IdentityOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.identity";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value size);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value size();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DiagonalOp
	//===----------------------------------------------------------------------===//

	class DiagonalOp;

	class DiagonalOpAdaptor : public OpAdaptor<DiagonalOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value values();
	};

	class DiagonalOp : public mlir::Op<DiagonalOp,
																		mlir::OpTrait::OneOperand,
																		mlir::OpTrait::OneResult,
																		mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = DiagonalOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.diagonal";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value values);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value values();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ZerosOp
	//===----------------------------------------------------------------------===//

	class ZerosOp;

	class ZerosOpAdaptor : public OpAdaptor<ZerosOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange sizes();
	};

	class ZerosOp : public mlir::Op<ZerosOp,
																 mlir::OpTrait::AtLeastNOperands<1>::Impl,
																 mlir::OpTrait::OneResult,
																 mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = ZerosOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.zeros";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange sizes);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange sizes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::OnesOp
	//===----------------------------------------------------------------------===//

	class OnesOp;

	class OnesOpAdaptor : public OpAdaptor<OnesOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange sizes();
	};

	class OnesOp : public mlir::Op<OnesOp,
																 mlir::OpTrait::AtLeastNOperands<1>::Impl,
																 mlir::OpTrait::OneResult,
																 mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = OnesOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.ones";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange sizes);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::ValueRange sizes();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::LinspaceOp
	//===----------------------------------------------------------------------===//

	class LinspaceOp;

	class LinspaceOpAdaptor : public OpAdaptor<LinspaceOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value start();
		mlir::Value end();
		mlir::Value steps();
	};

	class LinspaceOp : public mlir::Op<LinspaceOp,
																		mlir::OpTrait::NOperands<3>::Impl,
																		mlir::OpTrait::OneResult,
																		mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = LinspaceOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.linspace";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value start, mlir::Value end, mlir::Value steps);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value start();
		mlir::Value end();
		mlir::Value steps();
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

	class FillOp : public mlir::Op<FillOp,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = FillOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.fill";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value, mlir::Value memory);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value value();
		mlir::Value memory();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MinOp
	//===----------------------------------------------------------------------===//

	class MinOp;

	class MinOpAdaptor : public OpAdaptor<MinOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange values();
	};

	class MinOp : public mlir::Op<MinOp,
															 mlir::OpTrait::AtLeastNOperands<1>::Impl,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = MinOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.min";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange values);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::ValueRange values();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::MaxOp
	//===----------------------------------------------------------------------===//

	class MaxOp;

	class MaxOpAdaptor : public OpAdaptor<MaxOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::ValueRange values();
	};

	class MaxOp : public mlir::Op<MaxOp,
															 mlir::OpTrait::AtLeastNOperands<1>::Impl,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = MaxOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.max";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::ValueRange values);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::ValueRange values();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SumOp
	//===----------------------------------------------------------------------===//

	class SumOp;

	class SumOpAdaptor : public OpAdaptor<SumOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value array();
	};

	class SumOp : public mlir::Op<SumOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = SumOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.sum";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value array);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value array();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::ProductOp
	//===----------------------------------------------------------------------===//

	class ProductOp;

	class ProductOpAdaptor : public OpAdaptor<ProductOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value array();
	};

	class ProductOp : public mlir::Op<ProductOp,
																	 mlir::OpTrait::OneOperand,
																	 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = ProductOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.product";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value array);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		mlir::Type resultType();
		mlir::Value array();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::TransposeOp
	//===----------------------------------------------------------------------===//

	class TransposeOp;

	class TransposeOpAdaptor : public OpAdaptor<TransposeOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value matrix();
	};

	class TransposeOp : public mlir::Op<TransposeOp,
																		 mlir::OpTrait::OneOperand,
																		 mlir::OpTrait::OneResult,
																		 mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = TransposeOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.transpose";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value matrix);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value matrix();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::SymmetricOp
	//===----------------------------------------------------------------------===//

	class SymmetricOp;

	class SymmetricOpAdaptor : public OpAdaptor<SymmetricOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value matrix();
	};

	class SymmetricOp : public mlir::Op<SymmetricOp,
																			mlir::OpTrait::OneOperand,
																			mlir::OpTrait::OneResult,
																			mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = SymmetricOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.symmetric";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value matrix);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);
		mlir::LogicalResult verify();

		void getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects);

		mlir::Type resultType();
		mlir::Value matrix();
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

	class DerOp : public mlir::Op<DerOp,
															 mlir::OpTrait::OneOperand,
															 mlir::OpTrait::OneResult>
	{
		public:
		using Op::Op;
		using Adaptor = DerOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.der";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type resultType, mlir::Value operand);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Type resultType();
		mlir::Value operand();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::DerSeedOp
	//===----------------------------------------------------------------------===//

	class DerSeedOp;

	class DerSeedOpAdaptor : public OpAdaptor<DerSeedOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value member();
		unsigned int value();
	};

	class DerSeedOp : public mlir::Op<DerSeedOp,
																	 mlir::OpTrait::AtLeastNOperands<1>::Impl,
																	 mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = DerSeedOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.der_seed";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value member, unsigned int value);
		// TODO: static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value member();
		unsigned int value();
	};

	//===----------------------------------------------------------------------===//
	// Modelica::PrintOp
	//===----------------------------------------------------------------------===//

	class PrintOp;

	class PrintOpAdaptor : public OpAdaptor<PrintOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value value();
	};

	class PrintOp : public mlir::Op<PrintOp,
																 mlir::OpTrait::OneOperand,
																 mlir::OpTrait::ZeroResult>
	{
		public:
		using Op::Op;
		using Adaptor = PrintOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "modelica.print";
		}

		static llvm::ArrayRef<llvm::StringRef> getAttributeNames();
		static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value value);
		static mlir::ParseResult parse(mlir::OpAsmParser& parser, mlir::OperationState& result);
		void print(mlir::OpAsmPrinter& printer);

		mlir::Value value();
	};
}
