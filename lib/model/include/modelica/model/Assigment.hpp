#pragma once

#include "modelica/model/ModExp.hpp"

namespace modelica
{
	class InductionVar
	{
		public:
		InductionVar(size_t begin, size_t end): beginVal(begin), endVal(end)
		{
			assert(beginVal < endVal);	// NOLINT
		}
		InductionVar(size_t begin): beginVal(begin), endVal(begin + 1) {}

		[[nodiscard]] size_t begin() const { return beginVal; }
		[[nodiscard]] size_t end() const { return endVal; }

		private:
		size_t beginVal;
		size_t endVal;
	};

	class Assigment
	{
		public:
		Assigment(
				std::string name, ModExp exp, std::vector<InductionVar> inducts = {})
				: leftHand(std::move(name)),
					expression(std::move(exp)),
					inductionVars(std::move(inducts))
		{
		}
		[[nodiscard]] const std::string& getVarName() const { return leftHand; }

		[[nodiscard]] const ModExp& getExpression() const { return expression; }

		[[nodiscard]] size_t size() const { return inductionVars.size(); }
		[[nodiscard]] auto begin() { return inductionVars.begin(); }
		[[nodiscard]] auto begin() const { return inductionVars.begin(); }
		[[nodiscard]] auto end() { return inductionVars.end(); }
		[[nodiscard]] auto end() const { return inductionVars.end(); }
		[[nodiscard]] const InductionVar& getInductionVar(size_t index) const
		{
			return inductionVars.at(index);
		}

		private:
		std::string leftHand;
		ModExp expression;
		std::vector<InductionVar> inductionVars;
	};
}	 // namespace modelica
