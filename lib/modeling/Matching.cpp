#include "marco/modeling/Matching.h"

namespace marco::modeling::internal::matching
{
  Matchable::Matchable(MultidimensionalRange dimensions) : dimensions(std::move(dimensions))
  {
  }

  const MCIS& Matchable::getMatched() const
  {
    return match;
  }

  MCIS Matchable::getUnmatched() const
  {
    return match.complement(dimensions);
  }

  bool Matchable::allComponentsMatched() const
  {
    return match.contains(dimensions);
  }

  void Matchable::addMatch(const MCIS& newMatch)
  {
    match += newMatch;
  }

  void Matchable::removeMatch(const MCIS& removedMatch)
  {
    match -= removedMatch;
  }
}
