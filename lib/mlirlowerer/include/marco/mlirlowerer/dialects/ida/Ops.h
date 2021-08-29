#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "Type.h"

namespace marco::codegen::ida
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
	// Ida::ConstantValueOp
	//===----------------------------------------------------------------------===//

	class ConstantValueOp;

	class ConstantValueOpAdaptor : public OpAdaptor<ConstantValueOp>
	{
		public:
		using OpAdaptor::OpAdaptor;
	};

	class ConstantValueOp : public mlir::Op<ConstantValueOp,
																		mlir::OpTrait::ZeroRegion,
																		mlir::OpTrait::OneResult,
																		mlir::OpTrait::ZeroOperands,
																		mlir::OpTrait::ConstantLike>
	{
		public:
		using Op::Op;
		using Adaptor = ConstantValueOpAdaptor;

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
	// Ida::AllocIdaUserDataOp
	//===----------------------------------------------------------------------===//

	class AllocIdaUserDataOp;

	class AllocIdaUserDataOpAdaptor : public OpAdaptor<AllocIdaUserDataOp>
	{
		public:
		using OpAdaptor::OpAdaptor;

		mlir::Value neq();
		mlir::Value nnz();
	};

	class AllocIdaUserDataOp : public mlir::Op<AllocIdaUserDataOp,
																mlir::OpTrait::ZeroRegion,
																mlir::OpTrait::NOperands<2>::Impl,
																mlir::OpTrait::OneResult,
																mlir::MemoryEffectOpInterface::Trait>
	{
		public:
		using Op::Op;
		using Adaptor = AllocIdaUserDataOpAdaptor;

		static constexpr llvm::StringLiteral getOperationName()
		{
			return "ida.alloc_ida_user_data";
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
}
