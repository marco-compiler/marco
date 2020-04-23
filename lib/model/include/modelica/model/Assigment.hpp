#pragma once

#include <memory>

#include "modelica/model/ModEqTemplate.hpp"
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
				std::shared_ptr<ModEqTemplate> eq,
				MultiDimInterval inducts = {},
				bool forward = true)
				: body(std::move(eq)),

					inductionVars(std::move(inducts), forward)
		{
		}

		Assigment(
				ModExp left,
				ModExp exp,
				std::string name = "",
				MultiDimInterval inducts = {},
				bool forward = true)
				: body(std::make_unique<ModEqTemplate>(
							std::move(left), std::move(exp), std::move(name))),
					inductionVars(std::move(inducts), forward)
		{
		}
		Assigment(
				std::string left,
				ModExp exp,
				std::string name = "",
				MultiDimInterval inducts = {},
				bool forward = true)
				: body(std::make_shared<ModEqTemplate>(
							ModExp(std::move(left), exp.getModType()),
							std::move(exp),
							std::move(name))),
					inductionVars(std::move(inducts), forward)
		{
		}
		[[nodiscard]] const ModExp& getLeftHand() const { return body->getLeft(); }

		[[nodiscard]] const ModExp& getExpression() const
		{
			return body->getRight();
		}

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

			if (body->getName().empty())
			{
				getLeftHand().dump(OS);
				OS << " = ";
				getExpression().dump(OS);
			}
			else
			{
				OS << "template ";
				OS << body->getName();
			}

			OS << '\n';
		}

		[[nodiscard]] auto& getTemplate() { return body; }
		[[nodiscard]] const auto& getTemplate() const { return body; }

		private:
		std::shared_ptr<ModEqTemplate> body;
		OrderedMultiDimInterval inductionVars;
	};
}	 // namespace modelica
