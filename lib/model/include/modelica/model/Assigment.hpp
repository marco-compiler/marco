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
		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << '[' << begin() << ',' << end() << "]";
		}

		private:
		size_t beginVal;
		size_t endVal;
	};

	class Assigment
	{
		public:
		Assigment(
				ModExp left,
				ModExp exp,
				llvm::SmallVector<InductionVar, 3> inducts = {})
				: leftHand(std::move(left)),
					expression(std::move(exp)),
					inductionVars(std::move(inducts))
		{
		}
		Assigment(
				std::string left,
				ModExp exp,
				llvm::SmallVector<InductionVar, 3> inducts = {})
				: leftHand(std::move(left), exp.getModType()),
					expression(std::move(exp)),
					inductionVars(std::move(inducts))
		{
		}
		[[nodiscard]] const ModExp& getVarName() const { return leftHand; }

		[[nodiscard]] const ModExp& getExpression() const { return expression; }

		[[nodiscard]] size_t size() const { return inductionVars.size(); }
		[[nodiscard]] auto begin() { return inductionVars.begin(); }
		[[nodiscard]] auto begin() const { return inductionVars.begin(); }
		[[nodiscard]] auto end() { return inductionVars.end(); }
		[[nodiscard]] auto end() const { return inductionVars.end(); }
		[[nodiscard]] const llvm::SmallVector<InductionVar, 3>& getInductionVars()
				const
		{
			return inductionVars;
		}
		[[nodiscard]] const InductionVar& getInductionVar(size_t index) const
		{
			return inductionVars[index];
		}

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			if (!inductionVars.empty())
				OS << "for ";

			for (const auto& var : inductionVars)
				var.dump(OS);

			leftHand.dump(OS);
			OS << " = ";
			expression.dump(OS);

			OS << '\n';
		}

		private:
		ModExp leftHand;
		ModExp expression;
		llvm::SmallVector<InductionVar, 3> inductionVars;
	};
}	 // namespace modelica
