#include "marco/Dialect/Runtime/IR/Properties.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"

using namespace ::mlir::runtime;

namespace mlir::runtime {
PrintInfo::PrintInfo(bool value) : data(value) {}

PrintInfo::PrintInfo(IndexSet value) : data(std::move(value)) {}

bool PrintInfo::operator==(const PrintInfo &other) const {
  return data == other.data;
}

llvm::hash_code hash_value(const PrintInfo &printInfo) {
  if (printInfo.isa<bool>()) {
    return llvm::hash_value(printInfo.get<bool>());
  }

  return hash_value(printInfo.get<IndexSet>());
}

//===-------------------------------------------------------------------===//
// PrintableIndicesList
//===-------------------------------------------------------------------===//

mlir::LogicalResult setPropertiesFromAttribute(
    PrintableIndicesList &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr);

  if (!arrayAttr) {
    emitError() << "expected ArrayAttr to set VariablesList property";
    return mlir::failure();
  }

  for (mlir::Attribute elementAttr : arrayAttr) {
    if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(elementAttr)) {
      prop.emplace_back(boolAttr.getValue());
      continue;
    }

    IndexSet indices;

    if (mlir::failed(mlir::modeling::setPropertiesFromAttribute(
            indices, elementAttr, emitError))) {
      return mlir::failure();
    }

    prop.emplace_back(std::move(indices));
  }

  return mlir::success();
}

mlir::ArrayAttr getPropertiesAsAttribute(mlir::MLIRContext *context,
                                         const PrintableIndicesList &prop) {
  llvm::SmallVector<mlir::Attribute> attrs;
  mlir::Builder builder(context);

  for (const auto &printInfo : prop) {
    if (printInfo.isa<bool>()) {
      attrs.push_back(builder.getBoolAttr(printInfo.get<bool>()));
      continue;
    }

    attrs.push_back(mlir::modeling::getPropertiesAsAttribute(
        context, printInfo.get<IndexSet>()));
  }

  return builder.getArrayAttr(attrs);
}

llvm::hash_code computeHash(const PrintableIndicesList &prop) {
  return llvm::hash_combine_range(prop.begin(), prop.end());
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         PrintableIndicesList &prop) {
  uint64_t numOfElements;

  if (mlir::failed(reader.readVarInt(numOfElements))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < numOfElements; ++i) {
    uint64_t kind;

    if (mlir::failed(reader.readVarInt(kind))) {
      return mlir::failure();
    }

    if (kind == 0) {
      bool value;

      if (mlir::failed(reader.readBool(value))) {
        return mlir::failure();
      }

      prop.emplace_back(value);
    } else if (kind == 1) {
      IndexSet indices;

      if (mlir::failed(mlir::modeling::readFromMlirBytecode(reader, indices))) {
        return mlir::failure();
      }

      prop.emplace_back(std::move(indices));
    } else {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         const PrintableIndicesList &prop) {
  writer.writeVarInt(prop.size());

  for (const auto &printInfo : prop) {
    if (printInfo.isa<bool>()) {
      writer.writeVarInt(0);
      writer.writeOwnedBool(printInfo.get<bool>());
      continue;
    }

    writer.writeVarInt(1);

    mlir::modeling::writeToMlirBytecode(writer, printInfo.get<IndexSet>());
  }
}
} // namespace mlir::runtime
