#pragma once

#include <string>
#include <system_error>

#include "llvm/Support/Error.h"
#include "modelica/matching/Matching.hpp"
#include "modelica/model/EntryModel.hpp"

namespace modelica
{
	enum class MatchingErrorCode
	{
		success = 0,
		failedMatching,
		equationsAndStateMissmatch

	};
}

namespace std
{
	/**
	 * This class is required to specity that ParserErrorCode is a enum that is
	 * used to rappresent errors.
	 */
	template<>
	struct is_error_condition_enum<modelica::MatchingErrorCode>: public true_type
	{
	};
};	// namespace std

namespace modelica
{
	/**
	 * A category is required to be compatible with std::error.
	 * All this standard and has to be done this way, look for a
	 * guide regarding how to make error category if you need to change this.
	 *
	 * When you add a new error kind you must add that choise to message in the
	 * cpp file.
	 */
	class MatchingErrorCategory: public std::error_category
	{
		public:
		static MatchingErrorCategory category;
		[[nodiscard]] std::error_condition default_error_condition(int ev) const
				noexcept override;

		[[nodiscard]] const char* name() const noexcept override
		{
			return "Matching Error";
		}

		[[nodiscard]] bool equivalent(
				const std::error_code& code, int condition) const noexcept override;

		[[nodiscard]] std::string message(int ev) const noexcept override;
	};

	std::error_condition make_error_condition(MatchingErrorCode errc);

	class FailedMatching: public llvm::ErrorInfo<FailedMatching>
	{
		public:
		static char ID;
		FailedMatching(EntryModel model, size_t matchedCount)
				: model(std::move(model)), matchedCount(matchedCount)
		{
		}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Could only match " << matchedCount << " out of "
				 << model.equationsCount() << " in model: \n";
			model.dump(OS);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(MatchingErrorCode::failedMatching),
					MatchingErrorCategory::category);
		}

		private:
		EntryModel model;
		size_t matchedCount;
	};

	class EquationAndStateMissmatch
			: public llvm::ErrorInfo<EquationAndStateMissmatch>
	{
		public:
		static char ID;
		EquationAndStateMissmatch(EntryModel model): model(std::move(model)) {}

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "Could not match provided model: Eq count "
				 << model.equationsCount() << ", state count " << model.stateCount()
				 << "\n";
			model.dump(OS);
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(MatchingErrorCode::equationsAndStateMissmatch),
					MatchingErrorCategory::category);
		}

		private:
		EntryModel model;
	};

}	 // namespace modelica
