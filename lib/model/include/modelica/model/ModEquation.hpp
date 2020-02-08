#pragma once

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

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
			dumpInductions(OS);
			leftHand.dump(OS);
			OS << " = ";
			rightHand.dump(OS);
			OS << "\n";
		}

		void dumpInductions(llvm::raw_ostream& OS) const
		{
			for (const auto& ind : inductions)
				ind.dump(OS);
		}

		[[nodiscard]] bool isForEquation() const { return !inductions.empty(); }

		[[nodiscard]] IndexSet toIndexSet() const;
		llvm::Error explicitate(size_t argumentIndex, bool left);
		llvm::Error explicitate(const ModExpPath& path);
		void setInductionVars(llvm::SmallVector<InductionVar, 3> inds)
		{
			inductions = std::move(inds);
		}
		void setInductionVars(const MultiDimInterval& inds)
		{
			inductions = {};
			for (const auto& dim : inds)
				inductions.emplace_back(dim.min(), dim.max());
		}

		private:
		ModExp leftHand;
		ModExp rightHand;
		llvm::SmallVector<InductionVar, 3> inductions;
	};
}	 // namespace modelica
