#ifndef MARCO_DIAGNOSTIC_LOGMESSAGE_H
#define MARCO_DIAGNOSTIC_LOGMESSAGE_H

#include "marco/Diagnostic/Location.h"
#include "llvm/ADT/StringRef.h"

namespace marco::diagnostic
{
  class PrinterInstance;

  class Message
  {
    public:
      Message();

      Message(const Message& other);
      Message(Message&& other);

      virtual ~Message();

      Message& operator=(const Message& other);
      Message& operator=(Message&& other);

      virtual void print(PrinterInstance* printer) const = 0;
  };

	class SourceMessage : public Message
	{
		public:
      SourceMessage(SourceRange location);

      SourceMessage(const SourceMessage& other);
      SourceMessage(SourceMessage&& other);

      virtual ~SourceMessage();

      SourceMessage& operator=(const SourceMessage& other);
      SourceMessage& operator=(SourceMessage&& other);

    private:
      SourceRange location;
	};
}

#endif // MARCO_DIAGNOSTIC_LOGMESSAGE_H
