#include "marco/Modeling/Matching.h"

namespace marco::modeling::internal::matching
{
  Matchable::Matchable(IndexSet matchableIndices)
    : matchableIndices(std::move(matchableIndices))
  {
  }

  const IndexSet& Matchable::getMatched() const
  {
    return match;
  }

  IndexSet Matchable::getUnmatched() const
  {
    return matchableIndices - match;
  }

  bool Matchable::allComponentsMatched() const
  {
    return matchableIndices == match;
  }

  void Matchable::addMatch(const IndexSet& newMatch)
  {
    assert(matchableIndices.contains(newMatch) && "The matched indexes set is not a subset of the matchable indexes");
    match += newMatch;
  }

  void Matchable::removeMatch(const IndexSet& removedMatch)
  {
    match -= removedMatch;
  }
}
