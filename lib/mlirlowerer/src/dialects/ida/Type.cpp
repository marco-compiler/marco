#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <marco/mlirlowerer/dialects/ida/Type.h>

using namespace marco::codegen::ida;

BooleanType BooleanType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerType IntegerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

RealType RealType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

OpaquePointerType OpaquePointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerPointerType IntegerPointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

RealPointerType RealPointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

namespace marco::codegen::ida
{
	mlir::Type parseIdaType(mlir::DialectAsmParser& parser)
	{
		mlir::Builder builder = parser.getBuilder();

		if (mlir::succeeded(parser.parseOptionalKeyword("bool")))
			return BooleanType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("int")))
			return IntegerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("real")))
			return RealType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("opaque_ptr")))
			return OpaquePointerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("int_ptr")))
			return OpaquePointerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("real_ptr")))
			return OpaquePointerType::get(builder.getContext());

		parser.emitError(parser.getCurrentLocation()) << "unknown type";
		return mlir::Type();
	}

	void printIdaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
		llvm::raw_ostream& os = printer.getStream();

		if (type.isa<BooleanType>())
		{
			os << "bool";
			return;
		}

		if (type.isa<IntegerType>())
		{
			os << "int";
			return;
		}

		if (type.dyn_cast<RealType>())
		{
			os << "real";
			return;
		}

		if (type.isa<OpaquePointerType>())
		{
			os << "opaque_ptr";
			return;
		}

		if (type.isa<IntegerPointerType>())
		{
			os << "int_ptr";
			return;
		}

		if (type.isa<RealPointerType>())
		{
			os << "real_ptr";
			return;
		}

		os << "unknown type";
	}
}
