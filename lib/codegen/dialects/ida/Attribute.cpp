#include <marco/mlirlowerer/dialects/ida/Attribute.h>
#include <mlir/IR/DialectImplementation.h>

namespace marco::codegen::ida
{
	mlir::Attribute parseIdaAttribute(mlir::DialectAsmParser& parser, mlir::Type type)
	{
		return marco::codegen::modelica::parseModelicaAttribute(parser, type);
	}

	void printIdaAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer)
	{
		marco::codegen::modelica::printModelicaAttribute(attr, printer);
	}
}
