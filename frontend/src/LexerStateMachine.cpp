#include "modelica/LexerStateMachine.hpp"

#include <llvm/ADT/StringMap.h>
#include <map>
#include <tuple>

using namespace modelica;
using State = ModelicaStateMachine::State;

static bool isNonDigit(char c)
{
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
}

static char escapedChar(char c)
{
	constexpr int alert = 7;
	constexpr int backspace = 8;
	constexpr int fromFeed = 12;
	switch (c)
	{
		case ('a'):
			return static_cast<char>(alert);
		case ('b'):
			return static_cast<char>(backspace);
		case ('f'):
			return static_cast<char>(fromFeed);
		case ('n'):
			return '\n';
		case ('r'):
			return '\r';
		case ('t'):
			return '\t';
		case ('v'):
			return '\v';
		default:
			return c;
	}
}

static bool isDigit(char c) { return ('0' <= c && c <= '9'); }

static Token scanSymbol(char c)
{
	static const std::map<char, Token> symbols = []() {
		std::map<char, Token> symbols;

		symbols['*'] = Token::Multiply;
		symbols['-'] = Token::Minus;
		symbols['+'] = Token::Plus;
		symbols['/'] = Token::Division;
		symbols['.'] = Token::Dot;
		symbols['['] = Token::LSquare;
		symbols[']'] = Token::RSquare;
		symbols['{'] = Token::LCurly;
		symbols['}'] = Token::RCurly;
		symbols['('] = Token::LPar;
		symbols[')'] = Token::RPar;
		symbols['='] = Token::Equal;
		symbols['<'] = Token::LessThan;
		symbols['>'] = Token::GreaterThan;
		symbols[':'] = Token::Colons;

		return symbols;
	}();

	if (auto iter = symbols.find(c); iter != symbols.end())
		return iter->second;

	return Token::Error;
}

static Token stringToToken(const std::string& lookUp)
{
	static const llvm::StringMap<Token> keywords = []() {
		llvm::StringMap<Token> keywordMap;

		keywordMap["algorithm"] = Token::AlgorithmKeyword;
		keywordMap["and"] = Token::AndKeyword;
		keywordMap["annotation"] = Token::AnnotationKeyword;
		keywordMap["block"] = Token::BlockKeyword;
		keywordMap["break"] = Token::BreakKeyword;
		keywordMap["class"] = Token::ClassKeyword;
		keywordMap["connect"] = Token::ConnectKeyword;
		keywordMap["connector"] = Token::ConnectorKeyword;
		keywordMap["constant"] = Token::ConstantKeyword;
		keywordMap["constraynedby"] = Token::ConstraynedByKeyword;
		keywordMap["der"] = Token::DerKeyword;
		keywordMap["discrete"] = Token::DiscreteKeyword;
		keywordMap["each"] = Token::EachKeyword;
		keywordMap["else"] = Token::ElseKeyword;
		keywordMap["elseif"] = Token::ElseIfKeyword;
		keywordMap["elsewhen"] = Token::ElseWhenKeyword;
		keywordMap["encapsulated"] = Token::EncapsulatedKeyword;
		keywordMap["end"] = Token::EndKeyword;
		keywordMap["enumeration"] = Token::EnumerationKeyword;
		keywordMap["equation"] = Token::EquationKeyword;
		keywordMap["expandable"] = Token::ExpandableKeyword;
		keywordMap["extends"] = Token::ExtendsKeyword;
		keywordMap["external"] = Token::ExternalKeyword;
		keywordMap["false"] = Token::FalseKeyword;
		keywordMap["final"] = Token::FinalKeyword;
		keywordMap["flow"] = Token::FlowKeyword;
		keywordMap["for"] = Token::ForKeyword;
		keywordMap["function"] = Token::FunctionKeyword;
		keywordMap["if"] = Token::IfKeyword;
		keywordMap["import"] = Token::ImportKeyword;
		keywordMap["impure"] = Token::ImpureKeyword;
		keywordMap["in"] = Token::InKeyword;
		keywordMap["initial"] = Token::InitialKeyword;
		keywordMap["inner"] = Token::InnerKeyword;
		keywordMap["input"] = Token::InputKeyword;
		keywordMap["loop"] = Token::LoopKeyword;
		keywordMap["model"] = Token::ModelKeyword;
		keywordMap["not"] = Token::NotKeyword;
		keywordMap["operaptor"] = Token::OperaptorKeyword;
		keywordMap["or"] = Token::OrKeyword;
		keywordMap["outer"] = Token::OuterKeyword;
		keywordMap["output"] = Token::OutputKeyword;
		keywordMap["package"] = Token::PackageKeyword;
		keywordMap["parameter"] = Token::ParameterKeyword;
		keywordMap["partial"] = Token::PartialKeyword;
		keywordMap["protected"] = Token::ProtectedKeyword;
		keywordMap["public"] = Token::PublicKeyword;
		keywordMap["pure"] = Token::PureKeyword;
		keywordMap["record"] = Token::RecordKeyword;
		keywordMap["redeclare"] = Token::RedeclareKeyword;
		keywordMap["replacable"] = Token::ReplacableKeyword;
		keywordMap["return"] = Token::ReturnKeyword;
		keywordMap["strem"] = Token::StremKeyword;
		keywordMap["then"] = Token::ThenKeyword;
		keywordMap["true"] = Token::TrueKeyword;
		keywordMap["type"] = Token::TypeKeyword;
		keywordMap["when"] = Token::WhenKeyword;
		keywordMap["while"] = Token::WhileKeyword;
		keywordMap["whithin"] = Token::WhithinKeyword;

		return keywordMap;
	}();

	if (auto iter = keywords.find(lookUp); iter != keywords.end())
		return iter->getValue();

	return Token::Ident;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '*' && next == '/')
		state = State::EndOfComment;

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::EndOfComment>()
{
	state = State::Normal;
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingLineComment>()
{
	if (next == '\0')
		state = State::Normal;

	if (current == '\n')
		state = State::Normal;

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingNum>()
{
	if (isDigit(current))
		lastNum.addUpper(current - '0');

	if (current == '.')
	{
		state = State::ParsingFloat;
		return Token::None;
	}

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	auto isAccetable = [](char c) {
		return isDigit(c) || c == '.' || c == 'e' || c == 'E';
	};

	if (isAccetable(next))
		return Token::None;

	state = State::Normal;
	return Token::Integer;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloatExponent>()
{
	if (isDigit(current))
		lastNum.addExponential(current - '0');

	if (isDigit(next))
		return Token::None;

	state = State::Normal;
	return Token::FloatingPoint;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloatExponentialSign>()
{
	if (current == '-' || current == '+')
	{
		if (!isDigit(next))
		{
			error = "Exp sign must be followed by a number";
			state = State::Normal;
			return Token::Error;
		}
		state = State::ParsingFloatExponent;
		lastNum.setSign(current == '+');
		return Token::None;
	}

	if (isDigit(current))
	{
		state = State::ParsingFloatExponent;
		lastNum.addExponential(current - '0');
		return Token::None;
	}

	error = "Error unexpected char " + std::to_string(current) +
					" in floating number scan";
	state = State::Normal;
	return Token::Error;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingFloat>()
{
	if (isDigit(current))
		lastNum.addLower(current - '0');

	if (current == 'E' || current == 'e')
	{
		state = State::ParsingFloatExponentialSign;
		return Token::None;
	}

	if (isDigit(next) || next == 'E' || next == 'e')
		return Token::None;

	state = State::Normal;
	return Token::FloatingPoint;
}

template<>
Token ModelicaStateMachine::scan<State::Normal>()
{
	if (std::isspace(current) != 0)
		return Token::None;

	if (isNonDigit(current))
	{
		state = State::ParsingId;
		lastIdentifier = "";
		lastIdentifier.push_back(current);

		return Token::None;
	}

	if (current == '\'')
	{
		state = State::ParsingQId;
		lastIdentifier = "";
		return Token::None;
	}

	if (isDigit(current))
	{
		state = State::ParsingNum;
		lastNum = FloatLexer<defaultBase>();
		lastNum.addUpper(current - '0');
		return Token::None;
	}

	if (current == '/' && next == '/')
	{
		state = State::ParsingLineComment;
		return Token::None;
	}

	if (current == '/' && next == '*')
	{
		state = State::ParsingComment;
		return Token::None;
	}

	if (current == '"')
	{
		state = State::ParsingString;
		lastString = "";
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		return Token::End;
	}

	Token token = scanSymbol(current);
	if (token == Token::Error)
	{
		error = "Unexpeted character ";
		error.push_back(current);
	}

	return token;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingString>()
{
	if (current == '"')
	{
		state = State::Normal;
		return Token::String;
	}

	if (current == '\\')
	{
		state = State::ParsingBackSlash;
		return Token::None;
	}

	if (current == '\0')
	{
		state = State::End;
		error = "Reached end of string while parsing a string";
		return Token::Error;
	}

	lastString.push_back(current);
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingId>()
{
	lastIdentifier.push_back(current);

	if (!isDigit(next) && !isNonDigit(next))
	{
		state = State::Normal;
		return stringToToken(lastIdentifier);
	}

	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingQId>()
{
	if (current == '\\')
	{
		state = State::ParsingIdBackSlash;
		return Token::None;
	}

	if (current == '\'')
	{
		state = State::Normal;
		return Token::Ident;
	}

	if (next == '\0')
	{
		state = State::Normal;
		error = "unexpected end of string when parsing qidentifier";
		return Token::Error;
	}

	lastIdentifier.push_back(current);
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingIdBackSlash>()
{
	lastIdentifier.push_back(escapedChar(current));
	state = State::ParsingQId;
	return Token::None;
}

template<>
Token ModelicaStateMachine::scan<State::ParsingBackSlash>()
{
	lastString.push_back(escapedChar(current));
	state = State::ParsingString;
	return Token::None;
}

Token ModelicaStateMachine::step(char c)
{
	advance(c);
	switch (state)
	{
		case (Normal):
			return scan<Normal>();
		case (ParsingComment):
			return scan<ParsingComment>();
		case (ParsingLineComment):
			return scan<ParsingLineComment>();
		case (EndOfComment):
			return scan<EndOfComment>();
		case (ParsingNum):
			return scan<ParsingNum>();
		case (ParsingFloat):
			return scan<ParsingFloat>();
		case (ParsingFloatExponentialSign):
			return scan<ParsingFloatExponentialSign>();
		case (ParsingFloatExponent):
			return scan<ParsingFloatExponent>();
		case (ParsingString):
			return scan<ParsingString>();
		case (ParsingBackSlash):
			return scan<ParsingBackSlash>();
		case (ParsingId):
			return scan<ParsingId>();
		case (ParsingQId):
			return scan<ParsingQId>();
		case (ParsingIdBackSlash):
			return scan<ParsingIdBackSlash>();
		case (End):
			return Token::End;
	}

	error = "Unandled Lexer State";
	return Token::Error;
}
