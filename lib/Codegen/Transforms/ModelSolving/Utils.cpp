#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/MultidimensionalRange.h"
#include "llvm/ADT/StringSwitch.h"
#include <set>

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

static mlir::Attribute getSchedulingDirectionAttr(
    mlir::OpBuilder& builder,
    scheduling::Direction direction)
{
  if (direction == scheduling::Direction::None) {
    return builder.getStringAttr("none");
  }

  if (direction == scheduling::Direction::Forward) {
    return builder.getStringAttr("forward");
  }

  if (direction == scheduling::Direction::Backward) {
    return builder.getStringAttr("backward");
  }

  if (direction == scheduling::Direction::Constant) {
    return builder.getStringAttr("constant");
  }

  if (direction == scheduling::Direction::Mixed) {
    return builder.getStringAttr("mixed");
  }

  return builder.getStringAttr("unknown");
}

static scheduling::Direction getSchedulingDirection(mlir::Attribute attr)
{
  llvm::StringRef str = attr.cast<mlir::StringAttr>().getValue();

  return llvm::StringSwitch<scheduling::Direction>(str)
      .Case("none", scheduling::Direction::None)
      .Case("forward", scheduling::Direction::Forward)
      .Case("backward", scheduling::Direction::Backward)
      .Case("constant", scheduling::Direction::Constant)
      .Case("mixed", scheduling::Direction::Mixed)
      .Default(scheduling::Direction::Unknown);
}

namespace marco::codegen
{
  DerivativesMap readDerivativesMap(mlir::modelica::ModelOp modelOp)
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

  void writeDerivativesMap(mlir::OpBuilder& builder, mlir::modelica::ModelOp modelOp, const DerivativesMap& derivativesMap)
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

      matches.push_back(builder.getDictionaryAttr(namedAttrs));

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
          equations.add(std::move(matchedEquation));
        }
      }
    }

    result.setVariables(model.getVariables());
    result.setEquations(equations);
  }

  void writeSchedulingAttributes(mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model)
  {
    llvm::SmallVector<mlir::Attribute, 3> schedulingAttrs;

    for (const auto& equationsBlock : llvm::enumerate(model.getScheduledBlocks())) {
      for (const auto& equation : *equationsBlock.value()) {
        equation->getOperation()->removeAttr("schedule");
      }
    }

    for (const auto& equationsBlock : llvm::enumerate(model.getScheduledBlocks())) {
      for (const auto& equation : *equationsBlock.value()) {
        std::vector<mlir::Attribute> schedules;

        if (auto schedulesAttr = equation->getOperation()->getAttrOfType<mlir::ArrayAttr>("schedule")) {
          for (const auto& schedule : schedulesAttr) {
            schedules.push_back(schedule);
          }
        }

        std::vector<mlir::NamedAttribute> namedAttrs;

        namedAttrs.emplace_back(
            builder.getStringAttr("block"),
            builder.getI64IntegerAttr(equationsBlock.index()));

        namedAttrs.emplace_back(
            builder.getStringAttr("cycle"),
            builder.getBoolAttr(equationsBlock.value()->hasCycle()));

        namedAttrs.emplace_back(
            builder.getStringAttr("indices"),
            getIndexSetAttr(builder, equation->getIterationRanges()));

        namedAttrs.emplace_back(
            builder.getStringAttr("direction"),
            getSchedulingDirectionAttr(builder, equation->getSchedulingDirection()));

        schedules.push_back(builder.getDictionaryAttr(namedAttrs));

        mlir::Attribute newSchedulesAttr = builder.getArrayAttr(schedules);
        equation->getOperation()->setAttr("schedule", newSchedulesAttr);
      }
    }
  }

  void readSchedulingAttributes(const Model<MatchedEquation>& model, Model<ScheduledEquationsBlock>& result)
  {
    ScheduledEquationsBlocks scheduledEquationsBlocks;

    llvm::SmallVector<std::unique_ptr<ScheduledEquation>> scheduledEquations;
    llvm::DenseMap<int64_t, llvm::DenseSet<size_t>> blocks;
    llvm::DenseMap<int64_t, bool> cycles;

    for (const auto& equation : model.getEquations()) {
      if (auto scheduleAttr = equation->getOperation()->getAttrOfType<mlir::DictionaryAttr>("schedule")) {
        int64_t blockId = scheduleAttr.getAs<mlir::IntegerAttr>("block").getInt();
        IndexSet indices = getIndexSet(scheduleAttr.get("indices"));
        scheduling::Direction direction = getSchedulingDirection(scheduleAttr.get("direction"));

        auto matchedEquation = std::make_unique<MatchedEquation>(
            equation->clone(), equation->getIterationRanges(), equation->getWrite().getPath());

        auto scheduledEquation = std::make_unique<ScheduledEquation>(
                std::move(matchedEquation), indices, direction);

        scheduledEquations.push_back(std::move(scheduledEquation));
        blocks[blockId].insert(scheduledEquations.size() - 1);

        bool cycle = scheduleAttr.getAs<mlir::BoolAttr>("cycle").getValue();
        cycles[blockId] = cycle;
      }
    }

    std::vector<int64_t> orderedBlocksIds;

    for (const auto& block : blocks) {
      orderedBlocksIds.push_back(block.getFirst());
    }

    llvm::sort(orderedBlocksIds);

    for (const auto& blockId : orderedBlocksIds) {
      Equations<ScheduledEquation> equations;

      for (const auto& equationIndex : blocks[blockId]) {
        equations.push_back(std::move(scheduledEquations[equationIndex]));
      }

      auto scheduledEquationsBlock = std::make_unique<ScheduledEquationsBlock>(equations, cycles[blockId]);
      scheduledEquationsBlocks.append(std::move(scheduledEquationsBlock));
    }

    result.setVariables(model.getVariables());
    result.setScheduledBlocks(scheduledEquationsBlocks);
  }
}
