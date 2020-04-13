#pragma once

#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModExpPath.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/model/VectorAccess.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class ModEquation
	{
		public:
		ModEquation(
				ModExp left,
				ModExp right,
				MultiDimInterval inds = {},
				bool isForward = true)
				: leftHand(std::move(left)),
					rightHand(std::move(right)),
					inductions(std::move(inds)),
					isForCycle(!inductions.empty()),
					isForwardDirection(isForward)
		{
			if (!isForCycle)
				inductions = { { 0, 1 } };
		}

		[[nodiscard]] const ModExp& getLeft() const { return leftHand; }
		[[nodiscard]] const ModExp& getRight() const { return rightHand; }
		[[nodiscard]] ModExp& getLeft() { return leftHand; }
		[[nodiscard]] ModExp& getRight() { return rightHand; }
		[[nodiscard]] bool isForward() const { return isForwardDirection; }
		[[nodiscard]] const MultiDimInterval& getInductions() const
		{
			return inductions;
		}
		void foldConstants();

		void dump(llvm::raw_ostream& OS) const
		{
			if (!isForward())
				OS << "backward ";
			if (isForCycle)
			{
				OS << "for ";
				dumpInductions(OS);
			}
			leftHand.dump(OS);
			OS << " = ";
			rightHand.dump(OS);
			OS << "\n";
		}

		void dumpInductions(llvm::raw_ostream& OS) const { inductions.dump(OS); }

		[[nodiscard]] bool isForEquation() const { return isForCycle; }

		llvm::Error explicitate(size_t argumentIndex, bool left);
		llvm::Error explicitate(const ModExpPath& path);
		void setInductionVars(MultiDimInterval inds)
		{
			isForCycle = inds.empty();
			if (isForCycle)
				inductions = std::move(inds);
			else
				inds = { { 0, 1 } };
		}

		[[nodiscard]] AccessToVar getDeterminedVariable() const;
		[[nodiscard]] size_t dimensions() const
		{
			return isForCycle ? inductions.dimensions() : 0;
		}

		private:
		ModExp leftHand;
		ModExp rightHand;
		MultiDimInterval inductions;
		bool isForCycle;
		bool isForwardDirection;
	};
}	 // namespace modelica
