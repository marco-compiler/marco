#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <marco/mlirlowerer/dialects/ida/Type.h>

using namespace marco::codegen::ida;

OpaquePointerType OpaquePointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerPointerType IntegerPointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerType IntegerPointerType::getElementType() const
{
	return IntegerType::get(getContext());
}

RealPointerType RealPointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

RealType RealPointerType::getElementType() const
{
	return RealType::get(getContext());
}

namespace marco::codegen::ida
{
	mlir::Type parseIdaType(mlir::DialectAsmParser& parser)
	{
		mlir::Builder builder = parser.getBuilder();

		if (mlir::succeeded(parser.parseOptionalKeyword("opaque_ptr")))
			return OpaquePointerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("int_ptr")))
			return IntegerPointerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("real_ptr")))
			return RealPointerType::get(builder.getContext());

		return marco::codegen::modelica::parseModelicaType(parser);
	}

	void printIdaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
		llvm::raw_ostream& os = printer.getStream();

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

		marco::codegen::modelica::printModelicaType(type, printer);
	}
}
