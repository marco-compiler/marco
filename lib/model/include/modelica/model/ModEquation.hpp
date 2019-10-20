#pragma once

#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	class ModEquation
	{
		public:
		ModEquation(
				ModExp left, ModExp right, llvm::SmallVector<InductionVar, 3> inds = {})
				: leftHand(std::make_unique<ModExp>(std::move(left))),
					rightHand(std::make_unique<ModExp>(std::move(right))),
					inductions(std::move(inds))
		{
			assert(leftHand != nullptr);	 // NOLINT
			assert(rightHand != nullptr);	 // NOLINT
		}

		[[nodiscard]] const ModExp& getLeft() const { return *leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return *rightHand; }
		[[nodiscard]] ModExp& getLeft() { return *leftHand; }
		[[nodiscard]] ModExp& getRight() { return *rightHand; }
		[[nodiscard]] const auto& getInductions() const { return inductions; }
		[[nodiscard]] auto& getInductions() { return inductions; }

		void dump(llvm::raw_ostream& OS) const
		{
			if (!inductions.empty())
				OS << "for ";
			for (const auto& ind : inductions)
				ind.dump(OS);
			leftHand->dump(OS);
			OS << " = ";
			rightHand->dump(OS);
			OS << "\n";
		}

		private:
		std::unique_ptr<ModExp> leftHand;
		std::unique_ptr<ModExp> rightHand;
		llvm::SmallVector<InductionVar, 3> inductions;
	};
}	 // namespace modelica
