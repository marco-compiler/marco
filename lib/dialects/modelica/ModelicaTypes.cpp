#include "marco/dialects/modelica/ModelicaDialect.h"
#include "marco/dialects/modelica/ModelicaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

namespace mlir::modelica
{
  mlir::Type parseModelicaType(mlir::DialectAsmParser& parser)
  {
    auto builder = parser.getBuilder();

    if (mlir::succeeded(parser.parseOptionalKeyword("bool"))) {
      return BooleanType::get(builder.getContext());
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("int"))) {
      return IntegerType::get(builder.getContext());
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("real"))) {
      return RealType::get(builder.getContext());
    }

    /*
    if (mlir::succeeded(parser.parseOptionalKeyword("member")))
    {
      if (parser.parseLess())
        return mlir::Type();

      MemberAllocationScope scope;

      if (mlir::succeeded(parser.parseOptionalKeyword("stack")))
      {
        scope = MemberAllocationScope::stack;
      }
      else if (mlir::succeeded(parser.parseOptionalKeyword("heap")))
      {
        scope = MemberAllocationScope::heap;
      }
      else
      {
        parser.emitError(parser.getCurrentLocation()) << "unexpected member allocation scope";
        return mlir::Type();
      }

      if (parser.parseComma())
        return mlir::Type();

      llvm::SmallVector<int64_t, 3> dimensions;

      if (parser.parseDimensionList(dimensions))
        return mlir::Type();

      mlir::Type baseType;

      if (parser.parseType(baseType) ||
          parser.parseGreater())
        return mlir::Type();

      llvm::SmallVector<long, 3> castedDims(dimensions.begin(), dimensions.end());
      return MemberType::get(builder.getContext(), scope, baseType, castedDims);
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("array")))
    {
      if (parser.parseLess())
        return mlir::Type();

      if (mlir::succeeded(parser.parseOptionalStar()))
      {
        mlir::Type baseType;

        if (parser.parseType(baseType) ||
            parser.parseGreater())
          return mlir::Type();

        return UnsizedArrayType::get(builder.getContext(), baseType);
      }

      BufferAllocationScope scope = BufferAllocationScope::unknown;

      if (mlir::succeeded(parser.parseOptionalKeyword("stack")))
      {
        scope = BufferAllocationScope::stack;

        if (parser.parseComma())
          return mlir::Type();
      }
      else if (mlir::succeeded(parser.parseOptionalKeyword("heap")))
      {
        scope = BufferAllocationScope::heap;

        if (parser.parseComma())
          return mlir::Type();
      }

      llvm::SmallVector<int64_t, 3> dimensions;

      if (parser.parseDimensionList(dimensions))
        return mlir::Type();

      mlir::Type baseType;

      if (parser.parseType(baseType) ||
          parser.parseGreater())
        return mlir::Type();

      llvm::SmallVector<long, 3> castedDims(dimensions.begin(), dimensions.end());
      return ArrayType::get(builder.getContext(), scope, baseType, castedDims);
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("opaque_ptr")))
      return OpaquePointerType::get(builder.getContext());

    if (mlir::succeeded(parser.parseOptionalKeyword("struct")))
    {
      if (mlir::failed(parser.parseLess()))
        return mlir::Type();

      llvm::SmallVector<mlir::Type, 3> types;

      do {
        mlir::Type type;

        if (parser.parseType(type))
          return mlir::Type();

        types.push_back(type);
      } while (succeeded(parser.parseOptionalComma()));

      if (mlir::failed(parser.parseGreater()))
        return mlir::Type();

      return StructType::get(builder.getContext(), types);
    }
     */

    parser.emitError(parser.getCurrentLocation()) << "unknown type";
    return mlir::Type();
  }

  void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
    auto& os = printer.getStream();

    if (type.isa<BooleanType>()) {
      os << "bool";
      return;
    }

    if (type.isa<IntegerType>()) {
      os << "int";
      return;
    }

    if (type.dyn_cast<RealType>()) {
      os << "real";
      return;
    }

    /*
    if (auto memberType = type.dyn_cast<MemberType>())
    {
      os << "member<";

      if (memberType.getAllocationScope() == MemberAllocationScope::stack)
        os << "stack, ";
      else if (memberType.getAllocationScope() == MemberAllocationScope::heap)
        os << "heap, ";

      auto dimensions = memberType.getShape();

      for (const auto& dimension : dimensions)
        os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

      printer.printType(memberType.getElementType());
      os << ">";
      return;
    }

    if (auto arrayType = type.dyn_cast<ArrayType>())
    {
      os << "array<";

      if (arrayType.getAllocationScope() == BufferAllocationScope::stack)
        os << "stack, ";
      else if (arrayType.getAllocationScope() == BufferAllocationScope::heap)
        os << "heap, ";

      auto dimensions = arrayType.getShape();

      for (const auto& dimension : dimensions)
        os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

      printer.printType(arrayType.getElementType());
      os << ">";
      return;
    }

    if (auto arrayType = type.dyn_cast<UnsizedArrayType>())
    {
      os << "array<*x" << arrayType.getElementType() << ">";
      return;
    }

    if (type.isa<OpaquePointerType>())
    {
      os << "opaque_ptr";
      return;
    }

    if (auto structType = type.dyn_cast<StructType>())
    {
      os << "struct<";

      for (auto subtype : llvm::enumerate(structType.getElementTypes()))
      {
        if (subtype.index() != 0)
          os << ", ";

        os << subtype.value();
      }

      os << ">";
    }
    */
  }
}