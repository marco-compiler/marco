#pragma once

#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <mlir/IR/Types.h>

namespace marco::codegen::ida
{
	using BooleanType = marco::codegen::modelica::BooleanType;
	using IntegerType = marco::codegen::modelica::IntegerType;
	using RealType = marco::codegen::modelica::RealType;
	using ArrayType = marco::codegen::modelica::ArrayType;

	class OpaquePointerType : public mlir::Type::TypeBase<OpaquePointerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;
		static OpaquePointerType get(mlir::MLIRContext* context);
	};

	class IntegerPointerType : public mlir::Type::TypeBase<IntegerPointerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static IntegerPointerType get(mlir::MLIRContext* context);

		[[nodiscard]] IntegerType getElementType() const;
	};

	class RealPointerType : public mlir::Type::TypeBase<RealPointerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static RealPointerType get(mlir::MLIRContext* context);

		[[nodiscard]] RealType getElementType() const;
	};

	mlir::Type parseIdaType(mlir::DialectAsmParser& parser);
	void printIdaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}
