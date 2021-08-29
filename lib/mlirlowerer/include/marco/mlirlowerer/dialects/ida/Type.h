#pragma once

#include <mlir/IR/Types.h>

namespace marco::codegen::ida
{
	class IdaDialect;

	class BooleanType : public mlir::Type::TypeBase<BooleanType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static BooleanType get(mlir::MLIRContext* context);
	};

	class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static IntegerType get(mlir::MLIRContext* context);
	};

	class RealType : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static RealType get(mlir::MLIRContext* context);
	};

	class OpaquePointerType : public mlir::Type::TypeBase<OpaquePointerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;
		static OpaquePointerType get(mlir::MLIRContext* context);
	};

	mlir::Type parseIdaType(mlir::DialectAsmParser& parser);
	void printIdaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}
