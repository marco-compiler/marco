#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/Modeling/IR/Properties.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"

using namespace ::mlir::bmodelica;

namespace {
template <typename K, typename V>
bool mapsEqual(const llvm::MapVector<K, V> &map1,
               const llvm::MapVector<K, V> &map2) {
  if (map1.size() != map2.size()) {
    return false;
  }

  for (const auto &entry : map1) {
    auto it = map2.find(entry.first);

    if (it == map2.end() || it->second != entry.second) {
      return false;
    }
  }

  return true;
}
} // namespace

namespace mlir::bmodelica {
bool DerivativesMap::operator==(const DerivativesMap &other) const {
  return mapsEqual(derivatives, other.derivatives) &&
         mapsEqual(derivedIndices, other.derivedIndices) &&
         mapsEqual(inverseDerivatives, other.inverseDerivatives);
}

mlir::DictionaryAttr
DerivativesMap::asAttribute(mlir::MLIRContext *context) const {
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::NamedAttribute> namedAttrs;

  llvm::SmallVector<mlir::Attribute> derivativesAttrs;
  llvm::SmallVector<mlir::Attribute> derivedIndicesAttrs;
  llvm::SmallVector<mlir::Attribute> inverseDerivativesAttrs;

  for (const auto &entry : derivatives) {
    derivativesAttrs.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("variable", entry.first),
         builder.getNamedAttr("derivative", entry.second)}));
  }

  for (const auto &entry : derivedIndices) {
    auto indexSetAttr =
        mlir::modeling::getPropertiesAsAttribute(context, entry.second);

    derivedIndicesAttrs.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("variable", entry.first),
         builder.getNamedAttr("indices", indexSetAttr)}));
  }

  for (const auto &entry : inverseDerivatives) {
    inverseDerivativesAttrs.push_back(builder.getDictionaryAttr(
        {builder.getNamedAttr("derivative", entry.first),
         builder.getNamedAttr("variable", entry.second)}));
  }

  namedAttrs.push_back(builder.getNamedAttr(
      "derivatives", builder.getArrayAttr(derivativesAttrs)));

  namedAttrs.push_back(builder.getNamedAttr(
      "derivedIndices", builder.getArrayAttr(derivedIndicesAttrs)));

  namedAttrs.push_back(builder.getNamedAttr(
      "inverseDerivatives", builder.getArrayAttr(inverseDerivativesAttrs)));

  return builder.getDictionaryAttr(namedAttrs);
}

mlir::LogicalResult DerivativesMap::setFromAttr(
    DerivativesMap &prop, mlir::Attribute attr,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(attr);

  if (!dictAttr) {
    emitError() << "expected DictionaryAttr to set DerivativesMap property";
    return mlir::failure();
  }

  // Derivatives.
  auto derivativesAttrs = dictAttr.getAs<mlir::ArrayAttr>("derivatives");

  if (!derivativesAttrs) {
    emitError() << "expected ArrayAttr for key 'derivatives'";
    return mlir::failure();
  }

  for (mlir::Attribute entry : derivativesAttrs) {
    auto castedEntry = mlir::dyn_cast<mlir::DictionaryAttr>(entry);

    if (!castedEntry) {
      emitError() << "Expected DictionaryAttr for derivatives entry";
      return mlir::failure();
    }

    auto variable = castedEntry.getAs<mlir::SymbolRefAttr>("variable");

    if (!variable) {
      emitError() << "Expected SymbolRefAttr for derivatives key";
      return mlir::failure();
    }

    auto derivative = castedEntry.getAs<mlir::SymbolRefAttr>("derivative");

    if (!derivative) {
      emitError() << "Expected SymbolRefAttr for derivatives value";
      return mlir::failure();
    }

    prop.derivatives[variable] = derivative;
  }

  // Derived indices.
  auto derivedIndicesAttrs = dictAttr.getAs<mlir::ArrayAttr>("derivedIndices");

  if (!derivedIndicesAttrs) {
    emitError() << "expected ArrayAttr for key 'derivedIndices'";
    return mlir::failure();
  }

  for (mlir::Attribute entry : derivedIndicesAttrs) {
    auto castedEntry = mlir::dyn_cast<mlir::DictionaryAttr>(entry);

    if (!castedEntry) {
      emitError() << "Expected DictionaryAttr for derivedIndices entry";
      return mlir::failure();
    }

    auto variable = castedEntry.getAs<mlir::SymbolRefAttr>("variable");

    if (!variable) {
      emitError() << "Expected SymbolRefAttr for derivedIndices key";
      return mlir::failure();
    }

    mlir::modeling::IndexSet indices;

    if (mlir::failed(mlir::modeling::setPropertiesFromAttribute(
            indices, castedEntry.get("indices"), emitError))) {
      return mlir::failure();
    }

    prop.derivedIndices[variable] = indices;
  }

  // Inverse derivatives.
  auto inverseDerivativesAttrs =
      dictAttr.getAs<mlir::ArrayAttr>("inverseDerivatives");

  if (!inverseDerivativesAttrs) {
    emitError() << "expected ArrayAttr for key 'inverseDerivatives'";
    return mlir::failure();
  }

  for (mlir::Attribute entry : inverseDerivativesAttrs) {
    auto castedEntry = mlir::dyn_cast<mlir::DictionaryAttr>(entry);

    if (!castedEntry) {
      emitError() << "Expected DictionaryAttr for inverseDerivatives entry";
      return mlir::failure();
    }

    auto derivative = castedEntry.getAs<mlir::SymbolRefAttr>("derivative");

    if (!derivative) {
      emitError() << "Expected SymbolRefAttr for inverseDerivatives key";
      return mlir::failure();
    }

    auto variable = castedEntry.getAs<mlir::SymbolRefAttr>("variable");

    if (!variable) {
      emitError() << "Expected SymbolRefAttr for inverseDerivatives value";
      return mlir::failure();
    }

    prop.inverseDerivatives[derivative] = variable;
  }

  return mlir::success();
}

llvm::hash_code DerivativesMap::hash() const {
  return llvm::hash_combine(
      llvm::hash_combine_range(derivatives.begin(), derivatives.end()),
      llvm::hash_combine_range(derivedIndices.begin(), derivedIndices.end()),
      llvm::hash_combine_range(inverseDerivatives.begin(),
                               inverseDerivatives.end()));
}

mlir::LogicalResult readFromMlirBytecode(mlir::DialectBytecodeReader &reader,
                                         DerivativesMap &prop) {
  uint64_t derivativesCount;
  uint64_t derivedIndicesCount;
  uint64_t inverseDerivativesCount;

  if (mlir::failed(reader.readVarInt(derivativesCount))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < derivativesCount; ++i) {
    mlir::SymbolRefAttr first;
    mlir::SymbolRefAttr second;

    if (mlir::failed(reader.readAttribute(first))) {
      return mlir::failure();
    }

    if (mlir::failed(reader.readAttribute(second))) {
      return mlir::failure();
    }

    prop.derivatives[first] = second;
  }

  if (mlir::failed(reader.readVarInt(derivedIndicesCount))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < derivedIndicesCount; ++i) {
    mlir::SymbolRefAttr variable;
    mlir::modeling::IndexSet indices;

    if (mlir::failed(reader.readAttribute(variable))) {
      return mlir::failure();
    }

    if (mlir::failed(mlir::modeling::readFromMlirBytecode(reader, indices))) {
      return mlir::failure();
    }

    prop.derivedIndices[variable] = indices;
  }

  if (mlir::failed(reader.readVarInt(inverseDerivativesCount))) {
    return mlir::failure();
  }

  for (uint64_t i = 0; i < inverseDerivativesCount; ++i) {
    mlir::SymbolRefAttr first;
    mlir::SymbolRefAttr second;

    if (mlir::failed(reader.readAttribute(first))) {
      return mlir::failure();
    }

    if (mlir::failed(reader.readAttribute(second))) {
      return mlir::failure();
    }

    prop.inverseDerivatives[first] = second;
  }

  return mlir::success();
}

void writeToMlirBytecode(mlir::DialectBytecodeWriter &writer,
                         DerivativesMap &prop) {
  writer.writeVarInt(prop.derivatives.size());

  for (const auto &entry : prop.derivatives) {
    writer.writeAttribute(entry.first);
    writer.writeAttribute(entry.second);
  }

  writer.writeVarInt(prop.derivedIndices.size());

  for (const auto &entry : prop.derivedIndices) {
    writer.writeAttribute(entry.first);
    mlir::modeling::writeToMlirBytecode(writer, entry.second);
  }

  writer.writeVarInt(prop.inverseDerivatives.size());

  for (const auto &entry : prop.inverseDerivatives) {
    writer.writeAttribute(entry.first);
    writer.writeAttribute(entry.second);
  }
}

mlir::LogicalResult parse(mlir::OpAsmParser &parser, DerivativesMap &prop) {
  if (parser.parseLSquare()) {
    return mlir::failure();
  }

  if (mlir::failed(parser.parseOptionalRSquare())) {
    do {
      mlir::SymbolRefAttr variable;
      mlir::SymbolRefAttr derivative;

      if (parser.parseLess()) {
        return mlir::failure();
      }

      if (parser.parseCustomAttributeWithFallback(
              variable, parser.getBuilder().getType<mlir::NoneType>()) ||
          parser.parseComma() ||
          parser.parseCustomAttributeWithFallback(
              derivative, parser.getBuilder().getType<mlir::NoneType>())) {
        return mlir::failure();
      }

      prop.setDerivative(variable, derivative);

      if (mlir::succeeded(parser.parseOptionalComma())) {
        mlir::modeling::IndexSet indices;

        if (mlir::failed(mlir::modeling::parse(parser, indices))) {
          return mlir::failure();
        }

        prop.setDerivedIndices(variable, std::move(indices));
      } else {
        prop.setDerivedIndices(variable, {});
      }

      if (parser.parseGreater()) {
        return mlir::failure();
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (mlir::failed(parser.parseRSquare())) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void print(mlir::OpAsmPrinter &printer, const DerivativesMap &prop) {
  llvm::SmallVector<mlir::SymbolRefAttr> variables;

  for (const auto &entry : prop.derivatives) {
    variables.push_back(entry.first);
  }

  llvm::sort(variables, [](mlir::SymbolRefAttr first,
                           mlir::SymbolRefAttr second) {
    if (first.getRootReference() != second.getRootReference()) {
      return first.getRootReference().compare(second.getRootReference()) < 0;
    }

    size_t firstLength = first.getNestedReferences().size();
    size_t secondLength = second.getNestedReferences().size();
    size_t minLength = std::min(firstLength, secondLength);

    for (size_t i = 0; i < minLength; ++i) {
      mlir::FlatSymbolRefAttr firstRef = first.getNestedReferences()[i];
      mlir::FlatSymbolRefAttr secondRef = second.getNestedReferences()[i];

      if (firstRef != secondRef) {
        return firstRef.getValue().compare(secondRef.getValue()) < 0;
      }
    }

    return false;
  });

  printer << "[";

  llvm::interleaveComma(variables, printer, [&](mlir::SymbolRefAttr variable) {
    printer << "<" << variable << ", " << prop.getDerivative(variable);

    if (auto indices = prop.getDerivedIndices(variable);
        indices && !indices->get().empty()) {
      printer << ", ";
      mlir::modeling::print(printer, *indices);
    }

    printer << ">";
  });

  printer << "]";
}

llvm::SmallVector<mlir::SymbolRefAttr>
DerivativesMap::getDerivedVariables() const {
  llvm::SmallVector<mlir::SymbolRefAttr> result;
  result.reserve(derivatives.size());

  for (auto &entry : derivatives) {
    result.push_back(entry.first);
  }

  return result;
}

bool DerivativesMap::empty() const { return derivatives.empty(); }

std::optional<mlir::SymbolRefAttr>
DerivativesMap::getDerivative(mlir::SymbolRefAttr variable) const {
  auto it = derivatives.find(variable);

  if (it == derivatives.end()) {
    return std::nullopt;
  }

  return it->second;
}

/// Set the derivative variable for a state one.
void DerivativesMap::setDerivative(mlir::SymbolRefAttr variable,
                                   mlir::SymbolRefAttr derivative) {
  derivatives[variable] = derivative;
  inverseDerivatives[derivative] = variable;
}

std::optional<std::reference_wrapper<const marco::modeling::IndexSet>>
DerivativesMap::getDerivedIndices(mlir::SymbolRefAttr variable) const {
  auto it = derivedIndices.find(variable);
  assert(!getDerivative(variable) || it != derivedIndices.end());

  if (it == derivedIndices.end()) {
    return std::nullopt;
  }

  return std::reference_wrapper(it->second);
}

void DerivativesMap::setDerivedIndices(mlir::SymbolRefAttr variable,
                                       marco::modeling::IndexSet indices) {
  derivedIndices[variable] = std::move(indices);
}

void DerivativesMap::addDerivedIndices(mlir::SymbolRefAttr variable,
                                       marco::modeling::IndexSet indices) {
  derivedIndices[variable] += indices;
}

std::optional<mlir::SymbolRefAttr>
DerivativesMap::getDerivedVariable(mlir::SymbolRefAttr derivative) const {
  auto it = inverseDerivatives.find(derivative);

  if (it == inverseDerivatives.end()) {
    return std::nullopt;
  }

  return it->second;
}

LockedDerivativesMap::LockedDerivativesMap(DerivativesMap &derivativesMap)
    : derivativesMap(derivativesMap) {}

bool LockedDerivativesMap::empty() const {
  std::lock_guard lock(mutex);
  return derivativesMap.empty();
}

llvm::SmallVector<mlir::SymbolRefAttr>
LockedDerivativesMap::getDerivedVariables() const {
  std::lock_guard lock(mutex);
  return derivativesMap.getDerivedVariables();
}

std::optional<mlir::SymbolRefAttr>
LockedDerivativesMap::getDerivative(mlir::SymbolRefAttr variable) const {
  std::lock_guard lock(mutex);
  return derivativesMap.getDerivative(variable);
}

void LockedDerivativesMap::setDerivative(mlir::SymbolRefAttr variable,
                                         mlir::SymbolRefAttr derivative) {
  std::lock_guard lock(mutex);
  return derivativesMap.setDerivative(variable, derivative);
}

std::optional<std::reference_wrapper<const marco::modeling::IndexSet>>
LockedDerivativesMap::getDerivedIndices(mlir::SymbolRefAttr variable) const {
  std::lock_guard lock(mutex);
  return derivativesMap.getDerivedIndices(variable);
}

void LockedDerivativesMap::setDerivedIndices(
    mlir::SymbolRefAttr variable, marco::modeling::IndexSet indices) {
  std::lock_guard lock(mutex);
  return derivativesMap.setDerivedIndices(variable, std::move(indices));
}

void LockedDerivativesMap::addDerivedIndices(
    mlir::SymbolRefAttr variable, marco::modeling::IndexSet indices) {
  std::lock_guard lock(mutex);
  return derivativesMap.addDerivedIndices(variable, std::move(indices));
}

std::optional<mlir::SymbolRefAttr>
LockedDerivativesMap::getDerivedVariable(mlir::SymbolRefAttr derivative) const {
  std::lock_guard lock(mutex);
  return derivativesMap.getDerivedVariable(derivative);
}
} // namespace mlir::bmodelica
