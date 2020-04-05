#pragma once

#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class Assigment
	{
		public:
		Assigment(ModExp left, ModExp exp, MultiDimInterval inducts = {})
				: leftHand(std::move(left)),
					expression(std::move(exp)),
					inductionVars(std::move(inducts))
		{
		}
		Assigment(std::string left, ModExp exp, MultiDimInterval inducts = {})
				: leftHand(std::move(left), exp.getModType()),
					expression(std::move(exp)),
					inductionVars(std::move(inducts))
		{
		}
		[[nodiscard]] const ModExp& getLeftHand() const { return leftHand; }

		[[nodiscard]] const ModExp& getExpression() const { return expression; }

		[[nodiscard]] size_t size() const { return inductionVars.size(); }
		[[nodiscard]] auto begin() { return inductionVars.begin(); }
		[[nodiscard]] auto begin() const { return inductionVars.begin(); }
		[[nodiscard]] auto end() { return inductionVars.end(); }
		[[nodiscard]] auto end() const { return inductionVars.end(); }
		[[nodiscard]] const MultiDimInterval& getInductionVars() const
		{
			return inductionVars;
		}
		[[nodiscard]] MultiDimInterval& getInductionVars() { return inductionVars; }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			if (!inductionVars.empty())
				OS << "for ";

			inductionVars.dump(OS);

			leftHand.dump(OS);
			OS << " = ";
			expression.dump(OS);

			OS << '\n';
		}

		private:
		ModExp leftHand;
		ModExp expression;
		MultiDimInterval inductionVars;
	};
}	 // namespace modelica
