#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir::modelica;

//===----------------------------------------------------------------------===//
// Tablegen attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "marco/Dialect/Modelica/ModelicaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// ModelicaDialect
//===----------------------------------------------------------------------===//

namespace mlir::modelica
{
  void ModelicaDialect::registerAttributes()
  {
    addAttributes<
      #define GET_ATTRDEF_LIST
      #include "marco/Dialect/Modelica/ModelicaAttributes.cpp.inc"
    >();
  }
}

namespace mlir::modelica
{
  mlir::Attribute getAttr(mlir::Type type, long value)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), value > 0);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), value);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), value);
    }

    if (type.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(type, value);
    }

    if (type.isa<mlir::IntegerType>()) {
      return mlir::IntegerAttr::get(type, value);
    }

    if (type.isa<mlir::FloatType>()) {
      return mlir::FloatAttr::get(type, value);
    }

    llvm_unreachable("Unknown Modelica type");
    return {};
  }

  mlir::Attribute getAttr(mlir::Type type, double value)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), value > 0);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), value);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), value);
    }

    if (type.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(type, value);
    }

    if (type.isa<mlir::IntegerType>()) {
      return mlir::IntegerAttr::get(type, value);
    }

    if (type.isa<mlir::FloatType>()) {
      return mlir::FloatAttr::get(type, value);
    }

    llvm_unreachable("Unknown Modelica type");
    return {};
  }

  mlir::Attribute getZeroAttr(mlir::Type type)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), false);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), 0);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), 0);
    }

    if (type.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(type, 0);
    }

    llvm_unreachable("Unknown Modelica type");
    return {};
  }

  //===----------------------------------------------------------------------===//
  // BooleanAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute BooleanAttr::parse(mlir::AsmParser& parser, mlir::Type type)
  {
    bool value;

    if (parser.parseLess()) {
      return {};
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
      value = true;
    } else {
      if (parser.parseKeyword("false")) {
        return {};
      }

      value = false;
    }

    if (parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = BooleanType::get(parser.getContext());
    }

    return BooleanAttr::get(parser.getContext(), type, value);
  }

  void BooleanAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "<" << (getValue() == 0 ? "false" : "true") << ">";
  }

  //===----------------------------------------------------------------------===//
  // IntegerAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute IntegerAttr::parse(mlir::AsmParser& parser, mlir::Type type)
  {
    long value;

    if (parser.parseLess() ||
        parser.parseInteger(value) ||
        parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = IntegerType::get(parser.getContext());
    }

    return IntegerAttr::get(parser.getContext(), type, llvm::APInt(sizeof(long) * 8, value, true));
  }

  void IntegerAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "<" << getValue() << ">";
  }

  //===----------------------------------------------------------------------===//
  // RealAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute RealAttr::parse(mlir::AsmParser& parser, mlir::Type type)
  {
    double value;

    if (parser.parseLess() ||
        parser.parseFloat(value) ||
        parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = RealType::get(parser.getContext());
    }

    return RealAttr::get(parser.getContext(), type, llvm::APFloat(value));
  }

  void RealAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "<" << getValue() << ">";
  }

  //===----------------------------------------------------------------------===//
  // InverseFunctionsAttr
  //===----------------------------------------------------------------------===//

  bool InverseFunctionsMap::operator==(const InverseFunctionsMap& other) const
  {
    return map == other.map;
  }

  InverseFunctionsMap::InverseFunction& InverseFunctionsMap::operator[](unsigned int arg)
  {
    return map[arg];
  }

  bool InverseFunctionsMap::empty() const
  {
    return map.empty();
  }

  InverseFunctionsMap::iterator InverseFunctionsMap::begin()
  {
    return map.begin();
  }

  InverseFunctionsMap::const_iterator InverseFunctionsMap::begin() const
  {
    return map.begin();
  }

  InverseFunctionsMap::iterator InverseFunctionsMap::end()
  {
    return map.end();
  }

  InverseFunctionsMap::const_iterator InverseFunctionsMap::end() const
  {
    return map.end();
  }

  InverseFunctionsMap InverseFunctionsMap::allocateInto(mlir::StorageUniquer::StorageAllocator& allocator)
  {
    InverseFunctionsMap result;

    for (const auto& entry : map) {
      result.map[entry.first] = std::make_pair(
          allocator.copyInto(entry.second.first),
          allocator.copyInto(entry.second.second));
    }

    return result;
  }

  bool InverseFunctionsMap::isInvertible(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex) != map.end();
  }

  llvm::StringRef InverseFunctionsMap::getFunction(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex)->second.first;
  }

  llvm::ArrayRef<unsigned int> InverseFunctionsMap::getArgumentsIndexes(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex)->second.second;
  }

  llvm::hash_code hash_value(const InverseFunctionsMap& map) {
    return llvm::hash_combine_range(map.begin(), map.end());
  }

  mlir::Attribute InverseFunctionsAttr::parse(mlir::AsmParser& parser, mlir::Type)
  {
    // TODO parse InverseFunctionsAttr
    llvm_unreachable("InverseFunctionsAttr parsing is not implemented");
    return mlir::Attribute();
  }

  void InverseFunctionsAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "inverse<";
    auto inverseFunctionsMap = getInverseFunctionsMap();
    bool separator = false;

    for (auto invertibleArg : inverseFunctionsMap) {
      if (separator) {
        printer << ", ";
      }

      printer << invertibleArg << ": ";
      printer << inverseFunctionsMap.getFunction(invertibleArg) << "(";
      bool innerSeparator = false;

      for (const auto& arg : inverseFunctionsMap.getArgumentsIndexes(invertibleArg)) {
        if (innerSeparator) {
          printer << ", ";
        }

        printer << arg;
        innerSeparator = true;
      }

      printer << ")";
      separator = true;
    }

    printer << ">";
  }

  bool InverseFunctionsAttr::isInvertible(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().isInvertible(argumentIndex);
  }

  llvm::StringRef InverseFunctionsAttr::getFunction(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().getFunction(argumentIndex);
  }

  llvm::ArrayRef<unsigned int> InverseFunctionsAttr::getArgumentsIndexes(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().getArgumentsIndexes(argumentIndex);
  }

  //===----------------------------------------------------------------------===//
  // DerivativeAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute DerivativeAttr::parse(mlir::AsmParser& parser, mlir::Type type)
  {
    mlir::StringAttr name;
    unsigned int order = 1;

    if (parser.parseLess() ||
        parser.parseAttribute(name) ||
        parser.parseComma() ||
        parser.parseInteger(order) ||
        parser.parseGreater()) {
      return mlir::Attribute();
    }

    return DerivativeAttr::get(parser.getContext(), name.getValue(), order);
  }

  void DerivativeAttr::print(mlir::AsmPrinter& printer) const
  {
    printer << "derivative" << "<\"" << getName() << "\", " << getOrder() << ">";
  }
}
