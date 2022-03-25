#include "marco/Modeling/Matching.h"

namespace marco::modeling::internal::matching
{
  Matchable::Matchable(MultidimensionalRange dimensions) : dimensions(std::move(dimensions))
  {
  }

  const IndexSet& Matchable::getMatched() const
  {
    return match;
  }

  IndexSet Matchable::getUnmatched() const
  {
    return match.complement(dimensions);
  }

  bool Matchable::allComponentsMatched() const
  {
    return match.contains(dimensions);
  }

  void Matchable::addMatch(const IndexSet& newMatch)
  {
    [[maybe_unused]] auto containsIndexes = [&]() {
      IndexSet possibleIndexes(dimensions);
      return possibleIndexes.contains(newMatch);
    };

    assert(containsIndexes() && "The matched indexes set is not a subset of the matchable indexes");
    match += newMatch;
  }

  void Matchable::removeMatch(const IndexSet& removedMatch)
  {
    match -= removedMatch;
  }
}
