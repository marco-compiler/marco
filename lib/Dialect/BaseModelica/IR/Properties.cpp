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

Variable::Variable(const IndexSet &equationIndices,
                   const VariableAccess &access)
    : name(access.getVariable()),
      indices(access.getAccessFunction().map(equationIndices)) {}

bool Variable::operator==(const Variable &other) const {
  return name == other.name && indices == other.indices;
}

Variable::operator bool() const { return name != nullptr; }

mlir::Attribute Variable::asAttribute(mlir::MLIRContext *context) const {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  mlir::Builder builder(context);

  if (name) {
    attrs.push_back(builder.getNamedAttr("name", name));

    attrs.push_back(builder.getNamedAttr(
        "indices", mlir::modeling::getPropertiesAsAttribute(context, indices)));
  }

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

  prop.name = nameAttr;

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

//===-------------------------------------------------------------------===//
// Schedule
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    Schedule &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);

  if (!stringAttr) {
    return mlir::failure();
  }

  if (stringAttr == "any") {
    prop = Schedule::Any;
    return mlir::success();
  }

  if (stringAttr == "forward") {
    prop = Schedule::Forward;
    return mlir::success();
  }

  if (stringAttr == "backward") {
    prop = Schedule::Backward;
    return mlir::success();
  }

  if (stringAttr == "unknown") {
    prop = Schedule::Unknown;
    return mlir::success();
  }

  llvm_unreachable("Unknown schedule type");
  return mlir::success();
}

mlir::Attribute getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const Schedule &prop) {
  if (prop == Schedule::Any) {
    return mlir::StringAttr::get(context, "any");
  }

  if (prop == Schedule::Forward) {
    return mlir::StringAttr::get(context, "forward");
  }

  if (prop == Schedule::Backward) {
    return mlir::StringAttr::get(context, "backward");
  }

  if (prop == Schedule::Unknown) {
    return mlir::StringAttr::get(context, "unknown");
  }

  llvm_unreachable("Unknown schedule type");
  return {};
}

llvm::hash_code computeHash(const Schedule &prop) {
  return llvm::hash_value(prop);
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         Schedule &prop) {
  int64_t value;

  if (mlir::failed(reader.readSignedVarInt(value))) {
    return mlir::failure();
  }

  if (value == 0) {
    prop = Schedule::Unknown;
  } else if (value == 1) {
    prop = Schedule::Any;
  } else if (value == 2) {
    prop = Schedule::Forward;
  } else if (value == 3) {
    prop = Schedule::Backward;
  } else {
    return mlir::failure();
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const Schedule &prop) {
  if (prop == Schedule::Unknown) {
    writer.writeSignedVarInt(0);
  } else if (prop == Schedule::Any) {
    writer.writeSignedVarInt(1);
  } else if (prop == Schedule::Forward) {
    writer.writeSignedVarInt(2);
  } else if (prop == Schedule::Backward) {
    writer.writeSignedVarInt(3);
  } else {
    writer.writeSignedVarInt(-1);
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, Schedule &prop) {
  if (mlir::succeeded(parser.parseOptionalKeyword("any"))) {
    prop = Schedule::Any;
    return mlir::success();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("forward"))) {
    prop = Schedule::Forward;
    return mlir::success();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("backward"))) {
    prop = Schedule::Backward;
    return mlir::success();
  }

  if (mlir::succeeded(parser.parseKeyword("unknown"))) {
    prop = Schedule::Unknown;
    return mlir::success();
  }

  return mlir::failure();
}

void print(mlir::OpAsmPrinter &printer, const Schedule &prop) {
  if (prop == Schedule::Any) {
    printer << "any";
  } else if (prop == Schedule::Forward) {
    printer << "forward";
  } else if (prop == Schedule::Backward) {
    printer << "backward";
  } else if (prop == Schedule::Unknown) {
    printer << "unknown";
  } else {
    llvm_unreachable("Unknown schedule type");
  }
}

//===-------------------------------------------------------------------===//
// ScheduleList
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    ScheduleList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);

  if (!arrayAttr) {
    emitError() << "expected ArrayAttr to set ScheduleList property";
    return mlir::failure();
  }

  for (mlir::Attribute elementAttr : arrayAttr) {
    if (mlir::failed(setPropertiesFromAttribute(prop.emplace_back(),
                                                elementAttr, emitError))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const ScheduleList &prop) {
  llvm::SmallVector<mlir::Attribute> variableAttrs;

  for (const Schedule &schedule : prop) {
    variableAttrs.push_back(getPropertiesAsAttribute(context, schedule));
  }

  mlir::Builder builder(context);
  return builder.getArrayAttr(variableAttrs);
}

llvm::hash_code computeHash(const ScheduleList &prop) {
  return llvm::hash_combine_range(prop.begin(), prop.end());
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         ScheduleList &prop) {
  uint64_t numOfElements;

  if (mlir::failed(reader.readVarInt(numOfElements))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < numOfElements; ++i) {
    if (mlir::failed(readFromMlirBytecode(reader, prop.emplace_back()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const ScheduleList &prop) {
  writer.writeVarInt(prop.size());

  for (const Schedule &schedule : prop) {
    writeToMlirBytecode(writer, schedule);
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, ScheduleList &prop) {
  if (parser.parseLSquare()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRSquare())) {
    Schedule schedule;

    do {
      if (mlir::failed(parse(parser, schedule))) {
        return mlir::failure();
      }

      prop.push_back(std::move(schedule));
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const ScheduleList &prop) {
  printer << "[";

  llvm::interleaveComma(prop, printer, [&](const Schedule &variable) {
    print(printer, variable);
  });

  printer << "]";
}
} // namespace mlir::bmodelica
