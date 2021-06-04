#pragma once

#include <llvm/Support/Error.h>
#include <llvm/Support/WithColor.h>
#include <modelica/utils/SourcePosition.h>
#include <string>
#include <system_error>
#include <utility>

#include "LexerStateMachine.h"

namespace modelica::frontend
{
	class AbstractMessage
	{
		public:
		AbstractMessage();
		AbstractMessage(const AbstractMessage& other);
		AbstractMessage(AbstractMessage&& other);

		virtual ~AbstractMessage();

		AbstractMessage& operator=(const AbstractMessage& other);
		AbstractMessage& operator=(AbstractMessage&& other);

		[[nodiscard]] virtual SourceRange getLocation() const = 0;

		virtual bool printBeforeMessage(llvm::raw_ostream& os) const;
		virtual void printMessage(llvm::raw_ostream& os) const = 0;
		virtual bool printAfterLines(llvm::raw_ostream& os) const;

		[[nodiscard]] virtual std::function<void(llvm::raw_ostream&)> getFormatter() const = 0;

		void print(llvm::raw_ostream& os) const;
	};

	class ErrorMessage : public AbstractMessage
	{
		public:
		[[nodiscard]] std::function<void (llvm::raw_ostream &)> getFormatter() const override;
	};

	class WarningMessage : public AbstractMessage
	{
		public:
		[[nodiscard]] std::function<void (llvm::raw_ostream &)> getFormatter() const override;
	};
}

namespace modelica::frontend::detail
{
	enum class GenericErrorCode
	{
		success = 0,
		not_implemented,
		choice_not_found,
		empty_list,
		branches_types_do_not_match
	};
}

namespace std
{
	template<>
	struct is_error_condition_enum<modelica::frontend::detail::GenericErrorCode>
			: public std::true_type
	{
	};
}

namespace modelica::frontend
{
	namespace detail
	{
		class GenericErrorCategory: public std::error_category
		{
			public:
			static GenericErrorCategory category;

			[[nodiscard]] std::error_condition default_error_condition(int ev) const noexcept override;

			[[nodiscard]] const char* name() const noexcept override
			{
				return "Generic error";
			}

			[[nodiscard]] bool equivalent(
					const std::error_code& code, int condition) const noexcept override;

			[[nodiscard]] std::string message(int ev) const noexcept override;
		};

		std::error_condition make_error_condition(GenericErrorCode errc);
	}

	class NotImplemented : public llvm::ErrorInfo<NotImplemented>
	{
		public:
		static char ID;

		NotImplemented(llvm::StringRef message) : message(message.str())
		{
		}

		void log(llvm::raw_ostream& os) const override
		{
			os << message;
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::GenericErrorCode::not_implemented),
					detail::GenericErrorCategory::category);
		}

		private:
		std::string message;
	};

	class ChoiceNotFound: public llvm::ErrorInfo<ChoiceNotFound>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override { OS << "Choice Not Found"; }

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::GenericErrorCode::choice_not_found),
					detail::GenericErrorCategory::category);
		}
	};

	/**
	 * A list was recived empty when was expected not to be
	 */
	class EmptyList: public llvm::ErrorInfo<EmptyList>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "The list was expected not empty";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::GenericErrorCode::empty_list),
					detail::GenericErrorCategory::category);
		}
	};

	/**
	 * Branches do not have the same types.
	 */
	class BranchesTypeDoNotMatch: public llvm::ErrorInfo<BranchesTypeDoNotMatch>
	{
		public:
		static char ID;

		void log(llvm::raw_ostream& OS) const override
		{
			OS << "if else branches do not have the same type";
		}

		[[nodiscard]] std::error_code convertToErrorCode() const override
		{
			return std::error_code(
					static_cast<int>(detail::GenericErrorCode::branches_types_do_not_match),
					detail::GenericErrorCategory::category);
		}
	};
}
