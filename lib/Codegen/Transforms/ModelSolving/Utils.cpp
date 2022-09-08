#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/MultidimensionalRange.h"

using namespace marco::codegen;
using namespace marco::modeling;
using namespace mlir::modelica;

static mlir::Attribute getRangeAttr(
    mlir::MLIRContext* context,
    const MultidimensionalRange& multidimensionalRange)
{
  llvm::SmallVector<mlir::Attribute, 3> rangesAttrs;

  for (unsigned int i = 0, rank = multidimensionalRange.rank(); i < rank; ++i) {
    const auto& range = multidimensionalRange[i];

    std::vector<mlir::Attribute> boundaries;
    boundaries.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), range.getBegin()));
    boundaries.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), range.getEnd() - 1));

    rangesAttrs.push_back(mlir::ArrayAttr::get(context, boundaries));
  }

  return mlir::ArrayAttr::get(context, rangesAttrs);
}

static MultidimensionalRange getRange(mlir::Attribute attr)
{
  llvm::SmallVector<Range, 3> ranges;

  for (const auto& rangeAttr : attr.cast<mlir::ArrayAttr>()) {
    auto rangeArrayAttr = rangeAttr.cast<mlir::ArrayAttr>();

    ranges.emplace_back(
        rangeArrayAttr[0].cast<mlir::IntegerAttr>().getInt(),
        rangeArrayAttr[1].cast<mlir::IntegerAttr>().getInt());
  }

  return MultidimensionalRange(ranges);
}

namespace marco::codegen
{
  DerivativesMap getDerivativesMap(mlir::modelica::ModelOp modelOp)
  {
    DerivativesMap derivativesMap;
    auto derivativesAttr = modelOp->getAttrOfType<mlir::ArrayAttr>("derivatives");

    if (!derivativesAttr) {
      return derivativesMap;
    }

    auto variablesNames = modelOp.variableNames();
    llvm::DenseMap<llvm::StringRef, unsigned int> namesMap;

    for (const auto& name : llvm::enumerate(variablesNames)) {
      namesMap[name.value()] = name.index();
    }

    for (const auto& dict : derivativesAttr.getAsRange<mlir::DictionaryAttr>()) {
      auto variableAttr = dict.getAs<mlir::SymbolRefAttr>("variable");
      auto derivativeAttr = dict.getAs<mlir::SymbolRefAttr>("derivative");

      IndexSet indices;

      if (auto indicesAttr = dict.getAs<mlir::ArrayAttr>("indices")) {
        for (const auto& rangeAttr : indicesAttr) {
          indices += getRange(rangeAttr);
        }
      } else {
        indices += Point(0);
      }

      auto varIndex = namesMap[variableAttr.getLeafReference().getValue()];
      auto derIndex = namesMap[derivativeAttr.getLeafReference().getValue()];

      derivativesMap.setDerivative(varIndex, derIndex);
      derivativesMap.setDerivedIndices(varIndex, indices);
    }

    return derivativesMap;
  }

  void setDerivativesMap(mlir::modelica::ModelOp modelOp, const DerivativesMap& derivativesMap)
  {
    mlir::MLIRContext* context = modelOp.getContext();

    llvm::SmallVector<mlir::Attribute, 6> derivativeAttrs;
    auto variablesNames = modelOp.variableNames();
    auto members = mlir::cast<YieldOp>(modelOp.getVarsRegion().back().getTerminator()).getValues();

    for (const auto& variableName : llvm::enumerate(variablesNames)) {
      if (auto varIndex = variableName.index(); derivativesMap.hasDerivative(varIndex)) {
        auto derIndex = derivativesMap.getDerivative(varIndex);
        auto derivedIndices = derivativesMap.getDerivedIndices(varIndex);

        std::vector<mlir::NamedAttribute> namedAttrs;

        namedAttrs.emplace_back(
            mlir::StringAttr::get(context, "variable"),
            mlir::SymbolRefAttr::get(context, variableName.value()));

        namedAttrs.emplace_back(
            mlir::StringAttr::get(context, "derivative"),
            mlir::SymbolRefAttr::get(context, variablesNames[derIndex]));

        auto memberType = members[varIndex].getType().cast<MemberType>();

        if (memberType.hasRank()) {
          std::vector<mlir::Attribute> derivedIndicesAttrs;

          for (const auto& range : llvm::make_range(derivedIndices.begin(), derivedIndices.end())) {
            derivedIndicesAttrs.push_back(getRangeAttr(context, range));
          }

          namedAttrs.emplace_back(
              mlir::StringAttr::get(context, "indices"),
              mlir::ArrayAttr::get(context, derivedIndicesAttrs));
        }

        auto derivativeAttr = mlir::DictionaryAttr::get(context, namedAttrs);
        derivativeAttrs.push_back(derivativeAttr);
      }
    }

    auto derivativesAttr = mlir::ArrayAttr::get(modelOp.getContext(), derivativeAttrs);
    modelOp->setAttr("derivatives", derivativesAttr);
  }
}
