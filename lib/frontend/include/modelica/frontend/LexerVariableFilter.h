//
// Created by famiglia on 15/04/21.
//

#ifndef MODELICA_LEXERVARIABLEFILTER_H
#define MODELICA_LEXERVARIABLEFILTER_H

#include <iostream>
#include <regex>

#include "llvm/ADT/StringRef.h"

using namespace std;
using namespace llvm;

namespace modelica
{
	//	LEXER 	//

	// The lexer returns tokens [0-255] if it is an unknown character, otherwise
	// one
	// of these for known things.
	enum Token
	{
		tok_eof = -1,

		// commands
		tok_der = -2,

		// primary
		tok_identifier = -4,

	};

	static std::string IdentifierStr;	 // Filled in if tok_identifier
	static double NumVal;							 // Filled in if tok_number

	static bool isValidIDChar(char c)
	{
		return (isalnum(c) || (c == '(') || (c == ')') || (c == '_'));
	}
	static const char *inputString;
	static unsigned short i;
	/// gettok - Return the next token from standard input.
	static int gettok()
	{
		static int LastChar = ' ';	// current character

		// Skip any whitespace between tokens.
		while (isspace(LastChar))
			LastChar = inputString[i];
		i++;

		// VARIABLES AND DERIVATIVES
		if (isalpha(LastChar) || (LastChar == '_'))
		{	 // identifier: [a-zA-Z][a-zA-Z0-9]* or underscore
			IdentifierStr = LastChar;
			while (isValidIDChar(
					(LastChar = getchar())))	// keep reading until identifier stops
				IdentifierStr += LastChar;

			if (IdentifierStr == "der(")
				return tok_der;

			return tok_identifier;
		}
		return LastChar;
	}

	// === AST / PARSER === //

	/// ExprAST - Base class for all expression nodes.
	class ExprAST
	{
		public:
		virtual ~ExprAST() {}
	};

	/// NumberExprAST - Expression class for numeric literals like "1.0".
	class VariableExprAST: public ExprAST
	{
		string identifier;

		public:
		VariableExprAST(const string &identifier): identifier(identifier) {}
	};

	class DerivativeExprAST: public ExprAST
	{
		VariableExprAST derivedVariable;

		public:
		DerivativeExprAST(const VariableExprAST &derivedVariable)
				: derivedVariable(derivedVariable)
		{
		}
	};

	class RegexExprAST: public ExprAST
	{
		basic_regex<string> regex;

		public:
		RegexExprAST(const basic_regex<string> &regex): regex(regex) {}
	};

	/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the
	/// current
	/// token the parser is looking at.  getNextToken reads another token from
	/// the lexer and updates CurTok with its results.
	static int CurTok;
	static int getNextToken() { return CurTok = gettok(); }

	/// var ::= varID
	static std::unique_ptr<ExprAST> ParseVariableExpr()
	{
		std::string IdName = IdentifierStr;
		auto Result = std::make_unique<VariableExprAST>(IdName);
		getNextToken();	 // consume the number
		return std::move(Result);
	}

	static std::unique_ptr<ExprAST> ParseDerivative()
	{
		getNextToken();	 // eat "der("

		// create derivative
		std::string IdName = IdentifierStr;
		auto Result = std::make_unique<DerivativeExprAST>(IdName);
		getNextToken();	 // consume the identifier
		getNextToken();	 // eat ")"
		return std::move(Result);
		getNextToken();	 // eat ")"
	}
	
	class LexerVariableFilter
	{
		StringRef inputStringReference;



		// === CLASS IMPL ===== //

		public:
		LexerVariableFilter(const string &inputString)
				: inputStringReference(inputString)
		{
		}

		void split(void)
		{
			SmallVector<StringRef> elements;
			cout << "Splitting: " << inputStringReference.str() << endl;
			inputStringReference.split(elements, ',', -1, false);
			for (const auto &item : elements)
			{
				string ele = item.str();
				cout << ele << endl;

				while (true)
				{
					cout << "ready:";
					switch (CurTok)
					{
						case tok_eof:
							return;
						case tok_der:
							ParseDerivative();
							break;
						case tok_identifier:
							ParseVariableExpr();
							break;
						default:
							getNextToken();
							break;
					}
				}
			}
		};
	};
}	 // namespace modelica
#endif	// MODELICA_LEXERVARIABLEFILTER_H
