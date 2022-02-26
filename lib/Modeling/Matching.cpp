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
    match += newMatch;
  }

  void Matchable::removeMatch(const IndexSet& removedMatch)
  {
    match -= removedMatch;
  }
}
