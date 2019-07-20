#pragma once
#include <cmath>

namespace modelica
{
	constexpr int defaultBase = 10;

	/**
	 * A Integer lexer is a object that can be feed with
	 * integers a1, a2 ... aN and will build the integer
	 * (a1*base^N) + (a2*base^N-1) ... + (aN*base^1);
	 *
	 * Used to parse unsigned integer.
	 */
	template<int base = defaultBase>
	class IntegerLexer
	{
		public:
		/**
		 * Creates a new IntegerLexer loaded with the provided int.
		 */
		IntegerLexer(int startValue = 0): value(startValue) {}

		/**
		 * Adds the next element to the integer.
		 */
		IntegerLexer operator+=(int i)
		{
			value = value * base;
			value = value + i;
			return *this;
		}

		/**
		 * Returns the currently built value.
		 */
		[[nodiscard]] int get() const { return value; }

		private:
		int value;
	};

	/**
	 * Used to build floats in the form upper.lower * (base ^ exponent) as
	 * described by modelica specification.
	 *
	 */
	template<int base>
	class FloatLexer
	{
		public:
		/**
		 * Concatenate the number to the integer part.
		 */
		void addUpper(int i) { upperPart += i; }
		/**
		 * Concatenate the number to the rational part.
		 */
		void addLower(int i) { lowerPart += i; }

		/**
		 * Concatenate the number to the exponent part.
		 */
		void addExponential(int i)
		{
			hasExponential = true;
			exponential += i;
		}

		/**
		 * Set the exponent sign to + if true, - if false.
		 * Default is true.
		 */
		void setSign(bool sign)
		{
			hasExponential = true;
			expSign = sign;
		}

		/**
		 * Return the X part to make it compatible with IntLexer.
		 */
		[[nodiscard]] int getUpperPart() const { return upperPart.get(); }

		/**
		 * Returns upper.lower * (base ^ (sign * exponent))
		 */
		[[nodiscard]] double get() const
		{
			int mantissaNormalizer = 1;
			while (mantissaNormalizer < lowerPart.get())
				mantissaNormalizer *= base;

			double toReturn = upperPart.get();
			toReturn += double(lowerPart.get()) / mantissaNormalizer;

			int exp = exponential.get();
			if (!expSign)
				exp *= -1;

			if (hasExponential)
				toReturn *= std::pow(base, exp);

			return toReturn;
		}

		private:
		IntegerLexer<base> upperPart;
		IntegerLexer<base> lowerPart;
		bool hasExponential{ false };
		IntegerLexer<base> exponential;
		bool expSign{ true };
	};
}	// namespace modelica
