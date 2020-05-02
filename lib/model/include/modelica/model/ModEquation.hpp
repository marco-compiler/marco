#pragma once

#include <memory>
#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModEqTemplate.hpp"
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
				std::string templateName = "",
				MultiDimInterval inds = {},
				bool isForward = true);

		ModEquation(
				std::shared_ptr<ModEqTemplate> templ,
				MultiDimInterval interval,
				bool isForward)
				: body(std::move(templ)),
					inductions(std::move(interval)),
					isForCycle(!inductions.empty()),
					isForwardDirection(isForward)
		{
			if (!isForCycle)
				inductions = { { 0, 1 } };
		}

		[[nodiscard]] const ModExp& getLeft() const { return body->getLeft(); }
		[[nodiscard]] const ModExp& getRight() const { return body->getRight(); }
		[[nodiscard]] ModExp& getLeft() { return body->getLeft(); }
		[[nodiscard]] ModExp& getRight() { return body->getRight(); }
		[[nodiscard]] bool isForward() const { return isForwardDirection; }
		[[nodiscard]] const MultiDimInterval& getInductions() const
		{
			return inductions;
		}
		void foldConstants();

		void dump(llvm::raw_ostream& OS) const;

		void dumpInductions(llvm::raw_ostream& OS) const { inductions.dump(OS); }

		[[nodiscard]] bool isForEquation() const { return isForCycle; }

		llvm::Error explicitate(size_t argumentIndex, bool left);
		llvm::Error explicitate(const ModExpPath& path);
		void setInductionVars(MultiDimInterval inds);

		[[nodiscard]] AccessToVar getDeterminedVariable() const;
		[[nodiscard]] size_t dimensions() const
		{
			return isForCycle ? inductions.dimensions() : 0;
		}

		[[nodiscard]] ModEquation clone(std::string newName) const
		{
			ModEquation clone = *this;
			clone.body = std::make_shared<ModEqTemplate>(*body);
			clone.getTemplate()->setName(std::move(newName));
			return clone;
		}

		[[nodiscard]] const std::shared_ptr<ModEqTemplate>& getTemplate() const
		{
			return body;
		}

		[[nodiscard]] std::shared_ptr<ModEqTemplate>& getTemplate() { return body; }

		private:
		std::shared_ptr<ModEqTemplate> body;
		MultiDimInterval inductions;
		bool isForCycle;
		bool isForwardDirection;
	};
}	 // namespace modelica
