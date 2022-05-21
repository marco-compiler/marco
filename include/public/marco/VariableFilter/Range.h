#ifndef MARCO_VARIABLEFILTER_RANGE_H
#define MARCO_VARIABLEFILTER_RANGE_H

namespace marco::vf
{
	/// Represents an array range.
  class Range
  {
    public:
      static constexpr long kUnbounded = -1;

      Range(long lowerBound, long upperBound);

      bool hasLowerBound() const;
      long getLowerBound() const;

      bool hasUpperBound() const;
      long getUpperBound() const;

    private:
      long lowerBound, upperBound;
  };
}

#endif // MARCO_VARIABLEFILTER_RANGE_H
