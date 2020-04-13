#pragma once

#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class OrderedMultiDimInterval
	{
		public:
		explicit OrderedMultiDimInterval(
				MultiDimInterval interval = {}, bool isForward = true)
				: interval(std::move(interval)), forward(isForward)
		{
		}
		[[nodiscard]] bool isForward() const { return forward; }
		[[nodiscard]] bool isBackward() const { return !forward; }
		[[nodiscard]] MultiDimInterval& getInterval() { return interval; }
		[[nodiscard]] size_t dimensions() const { return interval.dimensions(); }
		[[nodiscard]] const MultiDimInterval& getInterval() const
		{
			return interval;
		}
		[[nodiscard]] Interval operator[](size_t index) const
		{
			return interval.at(index);
		}

		private:
		MultiDimInterval interval;
		bool forward;
	};

	class Assigment
	{
		public:
		Assigment(
				ModExp left,
				ModExp exp,
				MultiDimInterval inducts = {},
				bool forward = true)
				: leftHand(std::move(left)),
					expression(std::move(exp)),
					inductionVars(std::move(inducts), forward)
		{
		}
		Assigment(
				std::string left,
				ModExp exp,
				MultiDimInterval inducts = {},
				bool forward = true)
				: leftHand(std::move(left), exp.getModType()),
					expression(std::move(exp)),
					inductionVars(std::move(inducts), forward)
		{
		}
		[[nodiscard]] const ModExp& getLeftHand() const { return leftHand; }

		[[nodiscard]] const ModExp& getExpression() const { return expression; }

		[[nodiscard]] size_t size() const { return getInductionVars().size(); }
		[[nodiscard]] auto begin() { return getInductionVars().begin(); }
		[[nodiscard]] auto begin() const { return getInductionVars().begin(); }
		[[nodiscard]] auto end() { return getInductionVars().end(); }
		[[nodiscard]] auto end() const { return getInductionVars().end(); }
		[[nodiscard]] const MultiDimInterval& getInductionVars() const
		{
			return inductionVars.getInterval();
		}
		[[nodiscard]] const OrderedMultiDimInterval& getOrderedInductionsVar() const
		{
			return inductionVars;
		}

		[[nodiscard]] MultiDimInterval& getInductionVars()
		{
			return inductionVars.getInterval();
		}
		[[nodiscard]] bool isForward() const { return inductionVars.isForward(); }
		[[nodiscard]] bool isBackward() const { return inductionVars.isBackward(); }

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			if (inductionVars.isBackward())
				OS << "backward ";
			if (!getInductionVars().empty())
				OS << "for ";

			getInductionVars().dump(OS);

			leftHand.dump(OS);
			OS << " = ";
			expression.dump(OS);

			OS << '\n';
		}

		private:
		ModExp leftHand;
		ModExp expression;
		OrderedMultiDimInterval inductionVars;
	};
}	 // namespace modelica
