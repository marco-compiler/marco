#include "marco/Dialect/Modelica/OpInterfaces.h"

#include "marco/Dialect/Modelica/ModelicaOpInterfaces.cpp.inc"

namespace mlir::modelica
{
  uint64_t AdditionalInductions::addIterationSpace(IndexSet iterationSpace)
  {
    auto id = static_cast<uint64_t>(iterationSpaces.size());
    iterationSpaces.push_back(std::move(iterationSpace));
    return id;
  }

  uint64_t AdditionalInductions::getNumOfIterationSpaces() const
  {
    return iterationSpaces.size();
  }

  void AdditionalInductions::addInductionVariable(
      mlir::Value induction,
      uint64_t iterationSpace,
      uint64_t dimension)
  {
    assert(iterationSpace < getNumOfIterationSpaces());
    assert(dimension < iterationSpaces[iterationSpace].rank());
    inductions[induction] = std::make_pair(iterationSpace, dimension);
  }

  bool AdditionalInductions::hasInductionVariable(mlir::Value induction) const
  {
    return inductions.contains(induction);
  }

  const IndexSet& AdditionalInductions::getInductionSpace(
      mlir::Value induction) const
  {
    assert(hasInductionVariable(induction));
    auto it = inductions.find(induction);
    assert(it != inductions.end());
    return iterationSpaces[it->getSecond().first];
  }

  uint64_t AdditionalInductions::getInductionDimension(
      mlir::Value induction) const
  {
    assert(hasInductionVariable(induction));
    auto it = inductions.find(induction);
    assert(it != inductions.end());
    return it->getSecond().second;
  }

  const AdditionalInductions::Dependencies&
  AdditionalInductions::getInductionDependencies(mlir::Value induction) const
  {
    auto inductionsIt = inductions.find(induction);
    assert(inductionsIt != inductions.end());

    auto dependenciesIt = dependencies.find(inductionsIt->getSecond().first);
    assert(dependenciesIt != dependencies.end());

    const IterationSpaceDependencies& iterationSpaceDependencies =
        dependenciesIt->getSecond();

    auto iterationSpaceDependenciesIt =
        iterationSpaceDependencies.find(getInductionDimension(induction));

    assert(iterationSpaceDependenciesIt != iterationSpaceDependencies.end());
    return iterationSpaceDependenciesIt->getSecond();
  }

  void AdditionalInductions::addDimensionDependency(
      uint64_t iterationSpace, uint64_t dimension, uint64_t dependency)
  {
    assert(iterationSpace < getNumOfIterationSpaces());
    assert(dimension < iterationSpaces[iterationSpace].rank());
    assert(dependency < iterationSpaces[iterationSpace].rank());
    dependencies[iterationSpace][dimension].insert(dependency);
  }
}
