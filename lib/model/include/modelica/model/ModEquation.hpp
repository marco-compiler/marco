#pragma once

#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class ModEquation
	{
		public:
		ModEquation(
				ModExp left, ModExp right, llvm::SmallVector<InductionVar, 3> inds = {})
				: leftHand(std::move(left)),
					rightHand(std::move(right)),
					inductions(std::move(inds))
		{
		}

		[[nodiscard]] const ModExp& getLeft() const { return leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return rightHand; }
		[[nodiscard]] ModExp& getLeft() { return leftHand; }
		[[nodiscard]] ModExp& getRight() { return rightHand; }
		[[nodiscard]] const auto& getInductions() const { return inductions; }
		[[nodiscard]] auto& getInductions() { return inductions; }
		void foldConstants();

		void dump(llvm::raw_ostream& OS) const
		{
			if (!inductions.empty())
				OS << "for ";
			for (const auto& ind : inductions)
				ind.dump(OS);
			leftHand.dump(OS);
			OS << " = ";
			rightHand.dump(OS);
			OS << "\n";
		}

		[[nodiscard]] IndexSet toIndexSet() const;

		private:
		ModExp leftHand;
		ModExp rightHand;
		llvm::SmallVector<InductionVar, 3> inductions;
	};
}	 // namespace modelica
