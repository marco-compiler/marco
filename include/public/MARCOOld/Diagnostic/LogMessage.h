#ifndef MARCO_DIAGNOSTIC_LOGMESSAGE_H
#define MARCO_DIAGNOSTIC_LOGMESSAGE_H

#include <llvm/Support/raw_ostream.h>
#include <marco/Utils/SourcePosition.h>

/*
namespace marco::diagnostic
{
  class PrintInstance;

  class Message
  {
    public:
      Message();
      Message(const Message& other);
      Message(Message&& other);

      virtual ~Message();

      Message& operator=(const Message& other);
      Message& operator=(Message&& other);

      virtual void print(llvm::raw_ostream& os) const = 0;

      virtual void beforeMessage(const PrintInstance& print, llvm::raw_ostream& os) const;
      virtual void message(const PrintInstance& print, llvm::raw_ostream& os) const;
      virtual void afterMessage(const PrintInstance& print, llvm::raw_ostream& os) const;

      virtual void beforeMessage(llvm::raw_ostream& os) const;
      virtual void message(llvm::raw_ostream& os) const = 0;
      virtual void afterMessage(llvm::raw_ostream& os) const;
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

    void beforeMessage(const PrintInstance& print, llvm::raw_ostream& os) const override;
    void message(const PrintInstance& print, llvm::raw_ostream& os) const override;
    void afterMessage(const PrintInstance& print, llvm::raw_ostream& os) const override;

    private:
      SourceRange location;
	};
}
 */

#endif // MARCO_DIAGNOSTIC_LOGMESSAGE_H
