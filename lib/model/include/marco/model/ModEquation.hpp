#pragma once

#include <memory>
#include <optional>
#include <utility>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModEqTemplate.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModExpPath.hpp"
#include "marco/model/ModVariable.hpp"
#include "marco/model/VectorAccess.hpp"
#include "marco/utils/IndexSet.hpp"
#include "marco/utils/Interval.hpp"

namespace marco
{
	class ModEquation
	{
		public:
		ModEquation(
				ModExp left,
				ModExp right,
				std::string templateName = "",
				MultiDimInterval inds = {},
				bool isForward = true,
				std::optional<EquationPath> path = std::nullopt);

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

		void dump() const;
		void dump(llvm::raw_ostream& OS) const;

		void readableDump() const;
		void readableDump(llvm::raw_ostream& OS) const;

		void dumpInductions(llvm::raw_ostream& OS) const { inductions.dump(OS); }
		[[nodiscard]] bool isMatched() const { return matchedExpPath.has_value(); }
		[[nodiscard]] ModExp& getMatchedExp()
		{
			assert(isMatched());
			return reachExp(matchedExpPath.value());
		}
		[[nodiscard]] const ModExp& getMatchedExp() const
		{
			assert(isMatched());
			return reachExp(matchedExpPath.value());
		}

		[[nodiscard]] bool isForEquation() const { return isForCycle; }

		[[nodiscard]] ModExpPath getMatchedModExpPath() const
		{
			assert(isMatched());
			return ModExpPath(getMatchedExp(), *matchedExpPath);
		}

		[[nodiscard]] AccessToVar getDeterminedVariable() const
		{
			assert(isMatched());
			return AccessToVar::fromExp(getMatchedExp());
		}

		llvm::Error explicitate(size_t argumentIndex, bool left);
		llvm::Error explicitate(const ModExpPath& path);
		/**
		 * explicitate matched expression.
		 */
		llvm::Error explicitate()
		{
			auto error = explicitate(getMatchedModExpPath());
			if (error)
				return error;

			matchedExpPath = EquationPath({}, true);
			return llvm::Error::success();
		}

		void setInductionVars(MultiDimInterval inds);

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

		void setForward(bool isForward) { isForwardDirection = isForward; }

		/**
		 * return a copy of the equation so that the single variables access on the
		 * left side of the equation becomes indentity vector access.
		 */
		[[nodiscard]] llvm::Expected<ModEquation> normalized() const;

		[[nodiscard]] llvm::Expected<ModEquation> normalizeMatched() const;

		/**
		 * the induction range is multiplied by the transformation
		 * the vector access are modified to be equivalent
		 */
		[[nodiscard]] llvm::Expected<ModEquation> composeAccess(
				const VectorAccess& transformation) const;

		template<typename Path>
		[[nodiscard]] ModExp& reachExp(Path& path)
		{
			return path.isOnEquationLeftHand() ? path.reach(getLeft())
																				 : path.reach(getRight());
		}

		void setMatchedExp(EquationPath path);

		/**
		 * given a mod exp path returns the expression pointed
		 * by that path in this equation.
		 */
		template<typename Path>
		[[nodiscard]] const ModExp& reachExp(const Path& path) const
		{
			return path.isOnEquationLeftHand() ? path.reach(getLeft())
																				 : path.reach(getRight());
		}

		/**
		 * Tries to bring all the usages of the variable in the left hand of the
		 * equation to the left side of the equation.
		 */
		[[nodiscard]] ModEquation groupLeftHand() const;

		private:
		std::shared_ptr<ModEqTemplate> body;
		MultiDimInterval inductions;
		bool isForCycle;
		bool isForwardDirection;
		std::optional<EquationPath> matchedExpPath;
	};
}	 // namespace marco
