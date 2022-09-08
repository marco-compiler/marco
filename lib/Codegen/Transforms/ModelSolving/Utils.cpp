#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/MultidimensionalRange.h"

using namespace marco::codegen;
using namespace marco::modeling;
using namespace mlir::modelica;

static mlir::Attribute getRangeAttr(
    mlir::OpBuilder& builder,
    const MultidimensionalRange& multidimensionalRange)
{
  llvm::SmallVector<mlir::Attribute, 3> rangesAttrs;

  for (unsigned int i = 0, rank = multidimensionalRange.rank(); i < rank; ++i) {
    const auto& range = multidimensionalRange[i];

    std::vector<mlir::Attribute> boundaries;
    boundaries.push_back(builder.getI64IntegerAttr(range.getBegin()));
    boundaries.push_back(builder.getI64IntegerAttr(range.getEnd() - 1));

    rangesAttrs.push_back(builder.getArrayAttr(boundaries));
  }

  return builder.getArrayAttr(rangesAttrs);
}

static mlir::Attribute getIndexSetAttr(
    mlir::OpBuilder& builder,
    const IndexSet& indexSet)
{
  llvm::SmallVector<mlir::Attribute, 3> indices;

  for (const auto& range : llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
    indices.push_back(getRangeAttr(builder, range));
  }

  return builder.getArrayAttr(indices);
}

static MultidimensionalRange getRange(mlir::Attribute attr)
{
  llvm::SmallVector<Range, 3> ranges;

  for (const auto& rangeAttr : attr.cast<mlir::ArrayAttr>()) {
    auto rangeArrayAttr = rangeAttr.cast<mlir::ArrayAttr>();

    ranges.emplace_back(
        rangeArrayAttr[0].cast<mlir::IntegerAttr>().getInt(),
        rangeArrayAttr[1].cast<mlir::IntegerAttr>().getInt() + 1);
  }

  return MultidimensionalRange(ranges);
}

static IndexSet getIndexSet(mlir::Attribute attr)
{
  IndexSet result;

  for (const auto& rangeAttr : attr.cast<mlir::ArrayAttr>()) {
    result += getRange(rangeAttr);
  }

  return result;
}

static mlir::Attribute getMatchedPathAttr(
    mlir::OpBuilder& builder,
    const EquationPath& path)
{
  std::vector<mlir::Attribute> pathAttrs;

  if (path.getEquationSide() == EquationPath::LEFT) {
    pathAttrs.push_back(builder.getStringAttr("L"));
  } else {
    pathAttrs.push_back(builder.getStringAttr("R"));
  }

  for (const auto& index : path) {
    pathAttrs.push_back(builder.getIndexAttr(index));
  }

  return builder.getArrayAttr(pathAttrs);
}

static EquationPath getMatchedPath(mlir::Attribute attr)
{
  auto pathAttr = attr.cast<mlir::ArrayAttr>();
  std::vector<size_t> pathIndices;

  for (size_t i = 1; i < pathAttr.size(); ++i) {
    auto indexAttr = pathAttr[i].cast<mlir::IntegerAttr>();
    pathIndices.push_back(indexAttr.getInt());
  }

  auto sideAttr = pathAttr[0].cast<mlir::StringAttr>();

  if (sideAttr.getValue() == "L") {
    return EquationPath(EquationPath::LEFT, pathIndices);
  }

  return EquationPath(EquationPath::RIGHT, pathIndices);
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

  void setDerivativesMap(mlir::OpBuilder& builder, mlir::modelica::ModelOp modelOp, const DerivativesMap& derivativesMap)
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
            builder.getStringAttr("variable"),
            mlir::SymbolRefAttr::get(context, variableName.value()));

        namedAttrs.emplace_back(
            builder.getStringAttr("derivative"),
            mlir::SymbolRefAttr::get(context, variablesNames[derIndex]));

        auto memberType = members[varIndex].getType().cast<MemberType>();

        if (memberType.hasRank()) {
          namedAttrs.emplace_back(
              builder.getStringAttr("indices"),
              getIndexSetAttr(builder, derivedIndices));
        }

        derivativeAttrs.push_back(builder.getDictionaryAttr(namedAttrs));
      }
    }

    modelOp->setAttr("derivatives", builder.getArrayAttr(derivativeAttrs));
  }

  void writeMatchingAttributes(mlir::OpBuilder& builder, const Model<MatchedEquation>& model)
  {
    llvm::SmallVector<mlir::Attribute, 3> matchingAttrs;

    for (const auto& equation : model.getEquations()) {
      equation->getOperation()->removeAttr("match");
    }

    for (const auto& equation : model.getEquations()) {
      std::vector<mlir::Attribute> matches;

      if (auto matchesAttr = equation->getOperation()->getAttrOfType<mlir::ArrayAttr>("match")) {
        for (const auto& match : matchesAttr) {
          matches.push_back(match);
        }
      }

      std::vector<mlir::NamedAttribute> namedAttrs;

      namedAttrs.emplace_back(
          builder.getStringAttr("path"),
          getMatchedPathAttr(builder, equation->getWrite().getPath()));

      namedAttrs.emplace_back(
          builder.getStringAttr("indices"),
          getIndexSetAttr(builder, equation->getIterationRanges()));

      mlir::Attribute newMatchesAttr = builder.getArrayAttr(matches);
      equation->getOperation()->setAttr("match", newMatchesAttr);
    }
  }

  void readMatchingAttributes(const Model<Equation>& model, Model<MatchedEquation>& result)
  {
    Equations<MatchedEquation> equations;

    for (const auto& equation : model.getEquations()) {
      EquationInterface equationInt = equation->getOperation();

      if (auto matchAttr = equationInt->getAttrOfType<mlir::ArrayAttr>("match")) {
        for (const auto& match : matchAttr) {
          auto dict = match.cast<mlir::DictionaryAttr>();
          auto indices = getIndexSet(dict.get("indices"));
          auto path = getMatchedPath(dict.get("path"));

          auto matchedEquation = std::make_unique<MatchedEquation>(equation->clone(), indices, path);
        }
      }
    }

    result.setVariables(model.getVariables());
    result.setEquations(equations);
  }
}
