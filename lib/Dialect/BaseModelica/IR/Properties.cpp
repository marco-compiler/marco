#include "marco/Dialect/BaseModelica/IR/Properties.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica {
//===-------------------------------------------------------------------===//
// Variable
//===-------------------------------------------------------------------===//

Variable::Variable() = default;

Variable::Variable(mlir::SymbolRefAttr name, IndexSet indices)
    : name(name), indices(std::move(indices)) {}

bool Variable::operator==(const Variable &other) const {
  return name == other.name && indices == other.indices;
}

mlir::Attribute Variable::asAttribute(mlir::MLIRContext *context) const {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  mlir::Builder builder(context);
  attrs.push_back(builder.getNamedAttr("name", name));

  attrs.push_back(builder.getNamedAttr(
      "indices", mlir::modeling::getPropertiesAsAttribute(context, indices)));

  return builder.getDictionaryAttr(attrs);
}

mlir::LogicalResult Variable::setFromAttr(
    Variable &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);

  if (!dictAttr) {
    emitError() << "expected DictionaryAttr to set Variable property";
    return mlir::failure();
  }

  auto nameAttr = dictAttr.getAs<mlir::SymbolRefAttr>("name");

  if (!nameAttr) {
    emitError() << "expected SymbolRefAttr for key 'name'";
    return mlir::failure();
  }

  auto indicesAttr = dictAttr.get("indices");

  if (mlir::failed(::mlir::modeling::setPropertiesFromAttribute(
          prop.indices, indicesAttr, emitError))) {
    return mlir::failure();
  }

  return mlir::success();
}

llvm::hash_code Variable::hash() const {
  return llvm::hash_combine(name, indices);
}

llvm::hash_code hash_value(const Variable &value) { return value.hash(); }

mlir::LogicalResult
Variable::readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                               Variable &prop) {
  if (mlir::failed(reader.readAttribute(prop.name))) {
    return mlir::failure();
  }

  if (mlir::failed(
          mlir::modeling::readFromMlirBytecode(reader, prop.indices))) {
    return mlir::failure();
  }

  return mlir::success();
}

void Variable::writeToMlirBytecode(mlir::DialectBytecodeWriter &writer) const {
  writer.writeAttribute(name);
  mlir::modeling::writeToMlirBytecode(writer, indices);
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, Variable &prop) {
  if (mlir::succeeded(parser.parseOptionalLess())) {
    if (parser.parseCustomAttributeWithFallback(
            prop.name, parser.getBuilder().getType<mlir::NoneType>())) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
      if (mlir::failed(mlir::modeling::parse(parser, prop.indices))) {
        return mlir::failure();
      }
    }

    if (parser.parseGreater()) {
      return mlir::failure();
    }
  } else {
    if (parser.parseCustomAttributeWithFallback(
            prop.name, parser.getBuilder().getType<mlir::NoneType>())) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const Variable &prop) {
  if (!prop.indices.empty()) {
    printer << "<";
  }

  printer << prop.name;

  if (!prop.indices.empty()) {
    printer << ", ";
    mlir::modeling::print(printer, prop.indices);
  }

  if (!prop.indices.empty()) {
    printer << ">";
  }
}

//===-------------------------------------------------------------------===//
// VariablesList
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    VariablesList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);

  if (!arrayAttr) {
    emitError() << "expected ArrayAttr to set VariablesList property";
    return mlir::failure();
  }

  for (mlir::Attribute elementAttr : arrayAttr) {
    if (mlir::failed(Variable::setFromAttr(prop.emplace_back(), elementAttr,
                                           emitError))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const VariablesList &prop) {
  llvm::SmallVector<mlir::Attribute> variableAttrs;

  for (const Variable &variable : prop) {
    variableAttrs.push_back(variable.asAttribute(context));
  }

  mlir::Builder builder(context);
  return builder.getArrayAttr(variableAttrs);
}

llvm::hash_code computeHash(const VariablesList &prop) {
  return llvm::hash_combine_range(prop.begin(), prop.end());
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         VariablesList &prop) {
  uint64_t numOfElements;

  if (mlir::failed(reader.readVarInt(numOfElements))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < numOfElements; ++i) {
    if (mlir::failed(
            Variable::readFromMlirBytecode(reader, prop.emplace_back()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         VariablesList &prop) {
  writer.writeVarInt(prop.size());

  for (const Variable &variable : prop) {
    variable.writeToMlirBytecode(writer);
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, VariablesList &prop) {
  if (parser.parseLSquare()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRSquare())) {
    Variable variable;

    do {
      if (mlir::failed(parse(parser, variable))) {
        return mlir::failure();
      }

      prop.push_back(std::move(variable));
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const VariablesList &prop) {
  printer << "[";

  llvm::interleaveComma(prop, printer, [&](const Variable &variable) {
    print(printer, variable);
  });

  printer << "]";
}
} // namespace mlir::bmodelica
