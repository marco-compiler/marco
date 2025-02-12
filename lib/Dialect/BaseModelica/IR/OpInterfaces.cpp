#include "marco/Dialect/BaseModelica/IR/OpInterfaces.h"
#include "marco/Dialect/BaseModelica/IR/Attributes.h"

using namespace ::mlir::bmodelica;

#include "marco/Dialect/BaseModelica/IR/BaseModelicaOpInterfaces.cpp.inc"

namespace mlir::bmodelica {
uint64_t AdditionalInductions::addIterationSpace(IndexSet iterationSpace) {
  auto id = static_cast<uint64_t>(iterationSpaces.size());
  size_t rank = iterationSpace.rank();
  iterationSpaces.push_back(std::move(iterationSpace));
  dependencies[id] = IterationSpaceDependencies();

  for (size_t i = 0; i < rank; ++i) {
    dependencies[id][i] = {};
  }

  return id;
}

uint64_t AdditionalInductions::getNumOfIterationSpaces() const {
  return iterationSpaces.size();
}

void AdditionalInductions::addInductionVariable(mlir::Value induction,
                                                uint64_t iterationSpace,
                                                uint64_t dimension) {
  assert(iterationSpace < getNumOfIterationSpaces());
  assert(dimension < iterationSpaces[iterationSpace].rank());
  inductions[induction] = std::make_pair(iterationSpace, dimension);
}

bool AdditionalInductions::hasInductionVariable(mlir::Value induction) const {
  return inductions.contains(induction);
}

const IndexSet &
AdditionalInductions::getInductionSpace(mlir::Value induction) const {
  assert(hasInductionVariable(induction));
  auto it = inductions.find(induction);
  assert(it != inductions.end());
  return iterationSpaces[it->getSecond().first];
}

uint64_t
AdditionalInductions::getInductionDimension(mlir::Value induction) const {
  assert(hasInductionVariable(induction));
  auto it = inductions.find(induction);
  assert(it != inductions.end());
  return it->getSecond().second;
}

const AdditionalInductions::Dependencies &
AdditionalInductions::getInductionDependencies(mlir::Value induction) const {
  auto inductionsIt = inductions.find(induction);
  assert(inductionsIt != inductions.end());

  auto dependenciesIt = dependencies.find(inductionsIt->getSecond().first);
  assert(dependenciesIt != dependencies.end());

  const IterationSpaceDependencies &iterationSpaceDependencies =
      dependenciesIt->getSecond();

  auto iterationSpaceDependenciesIt =
      iterationSpaceDependencies.find(getInductionDimension(induction));

  assert(iterationSpaceDependenciesIt != iterationSpaceDependencies.end());
  return iterationSpaceDependenciesIt->getSecond();
}

void AdditionalInductions::addDimensionDependency(uint64_t iterationSpace,
                                                  uint64_t dimension,
                                                  uint64_t dependency) {
  assert(iterationSpace < getNumOfIterationSpaces());
  assert(dimension < iterationSpaces[iterationSpace].rank());
  assert(dependency < iterationSpaces[iterationSpace].rank());
  dependencies[iterationSpace][dimension].insert(dependency);
}
} // namespace mlir::bmodelica

namespace mlir::bmodelica::ad::forward {
State::State() = default;

State::State(State &&other) = default;

State::~State() = default;

State &State::operator=(State &&other) = default;

mlir::SymbolTableCollection &State::getSymbolTableCollection() {
  return symbolTableCollection;
}

void State::mapDerivative(mlir::Value original, mlir::Value mapped) {
  valueMapping.map(original, mapped);
}

void State::mapDerivatives(mlir::ValueRange original, mlir::ValueRange mapped) {
  assert(original.size() == mapped.size());
  valueMapping.map(original, mapped);
}

bool State::hasDerivative(mlir::Value original) const {
  return valueMapping.contains(original);
}

std::optional<mlir::Value> State::getDerivative(mlir::Value original) const {
  if (!hasDerivative(original)) {
    return std::nullopt;
  }

  return valueMapping.lookup(original);
}

void State::mapGenericOpDerivative(mlir::Operation *original,
                                   mlir::Operation *mapped) {
  generalOpMapping[original] = mapped;
}

bool State::hasGenericOpDerivative(mlir::Operation *original) const {
  return generalOpMapping.contains(original);
}

std::optional<mlir::Operation *>
State::getGenericOpDerivative(mlir::Operation *original) const {
  if (!hasGenericOpDerivative(original)) {
    return std::nullopt;
  }

  return generalOpMapping.lookup(original);
}
} // namespace mlir::bmodelica::ad::forward
