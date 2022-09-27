#ifndef MARCO_PARSER_FLOATLEXER_H
#define MARCO_PARSER_FLOATLEXER_H

#include "marco/Parser/IntegerLexer.h"
#include <cmath>

namespace marco
{
	/// Used to build floats in the form upper.lower * (base ^ exponent) as
	/// described by the Modelica specification.
  template<unsigned int Base>
  class FloatLexer
  {
    public:
      /// Concatenate the number to the integer part.
      void addUpper(int i)
      {
        upperPart += i;
      }

      /// Concatenate the number to the rational part.
      void addLower(int i)
      {
        lowerPart += i;

        if (i != 0) {
          leadingZero = false;
        }

        if (leadingZero) {
          ++fractionalLeadingZeros;
        }
      }

      /// Concatenate the number to the exponent part.
      void addExponential(int i)
      {
        hasExponential = true;
        exponential += i;
      }

      /// Set the exponent sign to + if true, - if false.
      void setSign(bool sign)
      {
        hasExponential = true;
        expSign = sign;
      }

      /// Return the X part to make it compatible with the IntegerLexer.
      int64_t getUpperPart() const
      {
        return upperPart.get();
      }

      /// Returns upper.lower * (base ^ (sign * exponent)).
      double get() const
      {
        int64_t mantissaNormalizer = 1;

        while (mantissaNormalizer <= lowerPart.get()) {
          mantissaNormalizer *= Base;
        }

        double toReturn = upperPart.get();
        toReturn += static_cast<double>(lowerPart.get()) / (mantissaNormalizer * std::pow(Base, fractionalLeadingZeros));

        auto exp = exponential.get();

        if (!expSign) {
          exp *= -1;
        }

        if (hasExponential) {
          toReturn *= std::pow(Base, exp);
        }

        return toReturn;
      }

      void reset()
      {
        upperPart.reset();
        lowerPart.reset();
        hasExponential = false;
        exponential.reset();
        expSign = true;
        leadingZero = true;
        fractionalLeadingZeros = 0;
      }

    private:
      IntegerLexer<Base> upperPart;
      IntegerLexer<Base> lowerPart;
      bool hasExponential = false;
      IntegerLexer<Base> exponential;
      bool expSign = true;
      bool leadingZero = true;
      int fractionalLeadingZeros = 0;
  };
}

#endif // MARCO_PARSER_FLOATLEXER_H
