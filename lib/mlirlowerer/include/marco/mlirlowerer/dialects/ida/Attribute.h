#pragma once

#include <marco/mlirlowerer/dialects/modelica/Attribute.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>

namespace marco::codegen::ida
{
	using BooleanAttribute = marco::codegen::modelica::BooleanAttribute;
	using IntegerAttribute = marco::codegen::modelica::IntegerAttribute;
	using RealAttribute = marco::codegen::modelica::RealAttribute;

	mlir::Attribute parseIdaAttribute(mlir::DialectAsmParser& parser, mlir::Type type);
	void printIdaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
