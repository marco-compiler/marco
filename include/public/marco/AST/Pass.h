#ifndef MARCO_AST_PASS_H
#define MARCO_AST_PASS_H

#include "marco/Diagnostic/Diagnostic.h"

namespace marco::ast
{
	class Class;

	class Pass
	{
		public:
      Pass(diagnostic::DiagnosticEngine& diagnostics);
      Pass(const Pass& other);

      Pass(Pass&& other);
      Pass& operator=(Pass&& other);

      virtual ~Pass();

      Pass& operator=(const Pass& other);

      virtual bool run(std::unique_ptr<Class>& cls) = 0;

    protected:
      diagnostic::DiagnosticEngine* diagnostics();

    private:
      diagnostic::DiagnosticEngine* diagnostics_;
	};
}

#endif // MARCO_AST_PASS_H
