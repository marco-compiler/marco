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

// DATA STRUCTUREs//
//============================================================================================================//

namespace modelica
{
	unordered_set<string> variables_ids;
	unordered_set<string> derivatives_ids;

	void throwError(void)
	{
		cout << "ERROR | generic." << endl;
		exit(-1);
	}

	//	LEXER 	//
	//============================================================================================================//
	// The lexer returns tokens [0-255] if it is an unknown character, otherwise
	// one
	// of these for known things.
	enum Token
	{
		tok_eof = -1,

		// commands
		tok_der = -2,
		tok_der_close = -3,

		// primary
		tok_identifier = -4,

		// array support
		tok_sq_bracket_l = -5,
		tok_sq_bracket_r = -6,
		tok_comma = -7,
		tok_colon = -8,
		tok_rng_value = -9

	};

	static std::string IdentifierStr;	 // Filled in if tok_identifier
	static const char *inputString;
	static unsigned short i = 0;
	static bool parsingArray;
	static string rangeValue;

	static bool isValidIDChar(char c)
	{
		return (isalnum(c) || (c == '(') || (c == ')') || (c == '_'));
	}

	/// gettok - Return the next token from the input string. T O K E N I Z E R
	static int gettok()
	{
		int LastChar = 32;	// start from white space 32 === ' '
		rangeValue = "";		// empty the range value buffer
		// Skip any whitespace between tokens.
		while (isspace(LastChar))
		{
			LastChar = (unsigned char) inputString[i];
			i++;
		}

		// VARIABLES AND DERIVATIVES (cannot start with a number)
		if (isalpha(LastChar) || (LastChar == '_'))
		{	 // identifier: [a-zA-Z][a-zA-Z0-9]* or underscore

			IdentifierStr = LastChar;

			// single variable derivative der(x)
			if ((inputString[i]) == ')')
			{
				return tok_identifier;
			}
			// identifier: [a-zA-Z][a-zA-Z0-9]* or underscore
			while (isValidIDChar((LastChar = inputString[i++])))
			{	 // keep reading until identifier stops
				IdentifierStr += LastChar;

				if (IdentifierStr == "der(")
					return tok_der;

				// check next character
				if ((inputString[i]) == ')')
				{
					return tok_identifier;
				}
				if ((inputString[i]) == '[')
				{
					parsingArray = true;
					return tok_identifier;
				}
				if (IdentifierStr == ")")
				{
					return tok_der_close;
				}
			}

			return tok_identifier;
		}

		switch (LastChar)
		{
			case ('['):
				return tok_sq_bracket_l;
			case (']'):
				return tok_sq_bracket_r;
			case (','):
				return tok_comma;
			case (':'):
				return tok_colon;
			case ('$'):
				rangeValue += (char) LastChar;
				return tok_rng_value;
			default:
				break;
		}

		// range value of an array
		if (isdigit(LastChar))
		{
			rangeValue += (char) LastChar;
			while (isdigit(LastChar = inputString[i]))
			{
				rangeValue += (char) LastChar;
				i++;
			}
			return tok_rng_value;
		}

		return LastChar;
	}

	// === AST / PARSER === //
	//============================================================================================================//

	/// ExprAST - Base class for all expression nodes.
	class ExprAST
	{
		public:
		virtual ~ExprAST() {}
	};

	class VariableExprAST: public ExprAST
	{
		string identifier;

		public:
		VariableExprAST(const string &identifier): identifier(identifier) {}

		string getIdentifier(void) { return identifier; }
	};

	class ArrayRangeAST: public ExprAST
	{
		const char rvalue, lvalue;

		public:
		ArrayRangeAST(const char rvalue, const char lvalue)
				: rvalue(rvalue), lvalue(lvalue)
		{
		}

		char getRvalue() const { return rvalue; }

		char getLvalue() const { return lvalue; }
	};

	class ArrayExprAST: public ExprAST
	{
		public:
		ArrayExprAST(
				const VariableExprAST &arrayVariableIdentifier,
				uint32_t dimension,
				ArrayRangeAST &ranges)
				: arrayVariableIdentifier(arrayVariableIdentifier),
					dimension(dimension),
					_ranges(ranges)
		{
		}

		private:
		// TODO numberOfRanges == dimension
		bool checkCorrectDimension(void) { return true; }

		VariableExprAST arrayVariableIdentifier;
		uint32_t dimension;
		ArrayRangeAST &_ranges;
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

	//===========================================ìGRAMMAR
	//PRODUCTIONS=================================================//
	//===============================================================================================================//
	/// var ::= string aAzZ09_
	static std::unique_ptr<VariableExprAST> ParseVariableExpr()
	{
		std::string idName = IdentifierStr;
		auto result = std::make_unique<VariableExprAST>(idName);
		// getNextToken();	 // consume the number
		return std::move(result);
	}

	static std::unique_ptr<ExprAST> ParseDerivative()
	{
		getNextToken();	 // eat "der("

		// create derivative
		std::string IdName = IdentifierStr;
		auto Result = std::make_unique<DerivativeExprAST>(IdName);
		getNextToken();	 // consume )
		return std::move(Result);
	}

	static std::unique_ptr<ExprAST> ParseArray()
	{
		// indices begin
		if (getNextToken() != tok_sq_bracket_l)
			throwError();	 //[
		cout << "\n opening bracket [";
		std::string arrayName = IdentifierStr;
		uint32_t dimension;

		std::vector<ArrayRangeAST> rangeVector;

		do
		{
			cout << "\n\tnew range:: ";
			if (getNextToken() != tok_rng_value)
				throwError();	 // left-value
			cout << "lval=" << rangeValue;

			if (getNextToken() != tok_colon)
				throwError();	 // :
			cout << " : ";
			if (getNextToken() != tok_rng_value)
				throwError();	 // right value
			cout << "l-val=" << rangeValue;

		} while (getNextToken() == tok_comma);

		// indices end

		if (CurTok != tok_sq_bracket_r)
			throwError();
		cout << " \nclosing bracket ]" << endl;

		cout << "\n the array has been parsed ";

		ArrayRangeAST ranges(3, 3);
		auto result = std::make_unique<ArrayExprAST>(arrayName, dimension, ranges);
	}

	class LexerVariableFilter
	{
		private:
		StringRef inputStringReference;

		void lexerReset(void)
		{
			CurTok = 0;
			i = 0;
			parsingArray = false;
		}
		// === CLASS IMPL ===== //
		// === CLASS IMPL ===== //

		public:
		LexerVariableFilter(const string &commandLineString)
				: inputStringReference(commandLineString)
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
				cout << "working on: ele" << endl;
				inputString = ele.c_str();
				bool flag = true;
				lexerReset();
				while (flag)
				{
					cout << "ready:";
					switch (CurTok)
					{
						case tok_eof:
							return;
						case tok_der:
							ParseDerivative();
							break;
						case tok_identifier: {
							if (parsingArray)
							{
								cout << "\n 🏁 parsing an array" << endl;
								unique_ptr<ExprAST> arrayNode = ParseArray();
							}

							cout << "parsing a variable identifier: ";
							unique_ptr<VariableExprAST> varExpr = ParseVariableExpr();
							variables_ids.insert(varExpr->getIdentifier());
							flag = false;
							// variable identifier parsed correctly
						}
						break;

						default:
							getNextToken();
							break;
					}
				}
			}
		};

		void splitTest(const string & test) {
			// SmallVector<StringRef> elements;
			// cout << "Splitting: " << inputStringReference.str() << endl;
			// inputStringReference.split(elements, ',', -1, false);
			inputStringReference = test;
			string ele = inputStringReference.str();
			cout << ele << endl;
			inputString = ele.c_str();
			bool flag = true;
			lexerReset();
			while (flag) {
				cout << "ready:";
				switch (CurTok) {
					case tok_eof:
						return;
					case tok_der:
						ParseDerivative();
						break;
					case tok_identifier: {

						if (parsingArray) {
							cout << "\n 🏁 parsing an array" << endl;
							unique_ptr<ExprAST> arrayNode = ParseArray();

							//TEMPORARY CODE
							exit(1);
						}

						cout << "parsing a variable identifier: ";
						unique_ptr<VariableExprAST> varExpr = ParseVariableExpr();
						variables_ids.insert(varExpr->getIdentifier());
						flag = false;
						// variable identifier parsed correctly
					} break;

					default:
						getNextToken();
						break;
				}
			}
		};
	};
}	 // namespace modelica
#endif	// MODELICA_LEXERVARIABLEFILTER_H
