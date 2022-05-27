#pragma once

#include "llvm/Support/raw_ostream.h"
#include "marco/Diagnostic/Location.h"

namespace marco
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
