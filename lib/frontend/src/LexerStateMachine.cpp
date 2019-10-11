#include "modelica/LexerStateMachine.hpp"

using namespace modelica;
using State = ModelicaStateMachine::State;

std::string modelica::tokenToString(Token token)
{
	switch (token)
	{
		case Token::None:
			return "None";
		case Token::Begin:
			return "Begin";
		case Token::Ident:
			return "Ident";
		case Token::Integer:
			return "Integer";
		case Token::FloatingPoint:
			return "FloatingPoint";
		case Token::String:
			return "String";
		case Token::Error:
			return "Error";

		case Token::AlgorithmKeyword:
			return "AlgorithmKeyword";
		case Token::AndKeyword:
			return "AndKeyword";
		case Token::AnnotationKeyword:
			return "AnnotationKeyword";
		case Token::BlockKeyword:
			return "BlockKeyword";
		case Token::BreakKeyword:
			return "BreakKeyword";
		case Token::ClassKeyword:
			return "ClassKeyword";
		case Token::ConnectKeyword:
			return "ConnectKeyword";
		case Token::ConnectorKeyword:
			return "ConnectorKeyword";
		case Token::ConstantKeyword:
			return "ConstantKeyword";
		case Token::ConstraynedByKeyword:
			return "ConstraynedByKeyword";
		case Token::DerKeyword:
			return "DerKeyword";
		case Token::DiscreteKeyword:
			return "DiscreteKeyword";
		case Token::EachKeyword:
			return "EachKeyword";
		case Token::ElseKeyword:
			return "ElseKeyword";
		case Token::ElseIfKeyword:
			return "ElseIfKeyword";
		case Token::ElseWhenKeyword:
			return "ElseWhenKeyword";
		case Token::EncapsulatedKeyword:
			return "EncapsulatedKeyword";
		case Token::EndKeyword:
			return "EndKeyword";
		case Token::EnumerationKeyword:
			return "EnumerationKeyword";
		case Token::EquationKeyword:
			return "EquationKeyword";
		case Token::ExpandableKeyword:
			return "ExpandableKeyword";
		case Token::ExtendsKeyword:
			return "ExtendsKeyword";
		case Token::ExternalKeyword:
			return "ExternalKeyword";
		case Token::FalseKeyword:
			return "FalseKeyword";
		case Token::FinalKeyword:
			return "FinalKeyword";
		case Token::FlowKeyword:
			return "FlowKeyword";
		case Token::ForKeyword:
			return "ForKeyword";
		case Token::FunctionKeyword:
			return "FunctionKeyword";
		case Token::IfKeyword:
			return "IfKeyword";
		case Token::ImportKeyword:
			return "ImportKeyword";
		case Token::ImpureKeyword:
			return "ImpureKeyword";
		case Token::InKeyword:
			return "InKeyword";
		case Token::InitialKeyword:
			return "InitialKeyword";
		case Token::InnerKeyword:
			return "InnerKeyword";
		case Token::InputKeyword:
			return "InputKeyword";
		case Token::LoopKeyword:
			return "LoopKeyword";
		case Token::ModelKeyword:
			return "ModelKeyword";
		case Token::NotKeyword:
			return "NotKeyword";
		case Token::OperatorKeyword:
			return "OperatorKeyword";
		case Token::OrKeyword:
			return "OrKeyword";
		case Token::OuterKeyword:
			return "OuterKeyword";
		case Token::OutputKeyword:
			return "OutputKeyword";
		case Token::PackageKeyword:
			return "PackageKeyword ";
		case Token::ParameterKeyword:
			return "ParameterKeyword";
		case Token::PartialKeyword:
			return "PartialKeyword";
		case Token::ProtectedKeyword:
			return "ProtectedKeyword";
		case Token::PublicKeyword:
			return "PublicKeyword ";
		case Token::PureKeyword:
			return "PureKeyword";
		case Token::RecordKeyword:
			return "RecordKeyword";
		case Token::RedeclareKeyword:
			return "RedeclareKeyword";
		case Token::ReplacableKeyword:
			return "ReplacableKeyword";
		case Token::ReturnKeyword:
			return "ReturnKeyword ";
		case Token::StremKeyword:
			return "StremKeyword";
		case Token::ThenKeyword:
			return "ThenKeyword";
		case Token::TrueKeyword:
			return "TrueKeyword";
		case Token::TypeKeyword:
			return "TypeKeyword";
		case Token::WhenKeyword:
			return "WhenKeyword";
		case Token::WhileKeyword:
			return "WhileKeyword";
		case Token::WhithinKeyword:
			return "WhithinKeyword";

		case Token::Multiply:
			return "Multiply";
		case Token::Division:
			return "Division";
		case Token::Dot:
			return "Dot";
		case Token::Plus:
			return "Plus";
		case Token::Minus:
			return "Minus";
		case Token::ElementWiseMinus:
			return "ElementWiseMinus";
		case Token::ElementWiseSum:
			return "ElementWiseSum";
		case Token::ElementWiseMultilpy:
			return "ElementWiseMultilpy";
		case Token::ElementWiseDivision:
			return "ElementWiseDivision";
		case Token::ElementWiseExponential:
			return "ElementWiseExponential";
		case Token::OperatorEqual:
			return "OperatorEqual";
		case Token::LessThan:
			return "LessThan";
		case Token::LessEqual:
			return "LessEqual";
		case Token::Equal:
			return "Equal";
		case Token::GreaterThan:
			return "GreaterThan";
		case Token::GreaterEqual:
			return "GreaterEqual";
		case Token::Different:
			return "Different";
		case Token::Colons:
			return "Colons";
		case Token::Semicolons:
			return "Semicolons";
		case Token::Comma:
			return "Comma ";
		case Token::LPar:
			return "LPar";
		case Token::RPar:
			return "RPar";
		case Token::LSquare:
			return "LSquare";
		case Token::RSquare:
			return "RSquare";
		case Token::LCurly:
			return "LCurly";
		case Token::RCurly:
			return "RCurly";
		case Token::Exponential:
			return "Exponential";
		case Token::Assignment:
			return "Assignment";

		case Token::End:
			return "End";
	}
	return "Unkown Token";
}

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

Token ModelicaStateMachine::charToToken(char c) const
{
	if (auto iter = symbols.find(c); iter != symbols.end())
		return iter->second;

	return Token::Error;
}

Token ModelicaStateMachine::stringToToken(const std::string& lookUp) const
{
	if (auto iter = keywordMap.find(lookUp); iter != keywordMap.end())
		return iter->getValue();

	return Token::Ident;
}

ModelicaStateMachine::ModelicaStateMachine(char first)
		: state(State::Normal),
			current('\0'),
			next(first),
			currentToken(Token::Begin),
			lastIdentifier(""),
			lastString(""),
			lineNumber(1),
			columnNumber(0)
{
	keywordMap["algorithm"] = Token::AlgorithmKeyword;
	keywordMap["and"] = Token::AndKeyword;
	keywordMap["annotation"] = Token::AnnotationKeyword;
	keywordMap["block"] = Token::BlockKeyword;
	keywordMap["break"] = Token::BreakKeyword;
	keywordMap["class"] = Token::ClassKeyword;
	keywordMap["connect"] = Token::ConnectKeyword;
	keywordMap["connector"] = Token::ConnectorKeyword;
	keywordMap["constant"] = Token::ConstantKeyword;
	keywordMap["constrainedby"] = Token::ConstraynedByKeyword;
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
	keywordMap["operator"] = Token::OperatorKeyword;
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
	keywordMap["replaceable"] = Token::ReplacableKeyword;
	keywordMap["return"] = Token::ReturnKeyword;
	keywordMap["strem"] = Token::StremKeyword;
	keywordMap["then"] = Token::ThenKeyword;
	keywordMap["true"] = Token::TrueKeyword;
	keywordMap["type"] = Token::TypeKeyword;
	keywordMap["when"] = Token::WhenKeyword;
	keywordMap["while"] = Token::WhileKeyword;
	keywordMap["whithin"] = Token::WhithinKeyword;

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
	symbols[';'] = Token::Semicolons;
	symbols[','] = Token::Comma;
	symbols['^'] = Token::Exponential;
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

static Token elementWise(char current, char next)
{
	if (current != '.')
		return Token::None;
	switch (next)
	{
		case ('/'):
			return Token::ElementWiseDivision;
		case ('*'):
			return Token::ElementWiseMultilpy;
		case ('-'):
			return Token::ElementWiseMinus;
		case ('+'):
			return Token::ElementWiseSum;
		case ('^'):
			return Token::ElementWiseExponential;
	}
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

	if (!isAccetable(next))
	{
		state = State::Normal;
		return Token::Integer;
	}

	return Token::None;
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
Token ModelicaStateMachine::scan<State::Normal>()
{
	if (std::isspace(current) != 0)
		return Token::None;

	if (isNonDigit(current))
	{
		state = State::ParsingId;
		lastIdentifier = "";

		return scan<State::ParsingId>();
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
		return scan<State::ParsingNum>();
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

	return tryScanSymbol();
}

Token ModelicaStateMachine::tryScanSymbol()
{
	state = State::IgnoreNextChar;
	if (current == '<' && next == '>')
		return Token::Different;

	if (current == '<' && next == '=')
		return Token::LessEqual;

	if (current == '>' && next == '=')
		return Token::GreaterEqual;

	if (next == '=' && current == '=')
		return Token::OperatorEqual;

	if (current == ':' && next == '=')
		return Token::Assignment;

	if (Token token = elementWise(current, next); token != Token::None)
		return token;

	state = State::Normal;
	Token token = charToToken(current);
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
		case (State::Normal):
			return scan<State::Normal>();
		case (State::ParsingComment):
			return scan<State::ParsingComment>();
		case (State::ParsingLineComment):
			return scan<State::ParsingLineComment>();
		case (State::EndOfComment):
			return scan<State::EndOfComment>();
		case (State::ParsingNum):
			return scan<State::ParsingNum>();
		case (State::ParsingFloat):
			return scan<State::ParsingFloat>();
		case (State::ParsingFloatExponentialSign):
			return scan<State::ParsingFloatExponentialSign>();
		case (State::ParsingFloatExponent):
			return scan<State::ParsingFloatExponent>();
		case (State::ParsingString):
			return scan<State::ParsingString>();
		case (State::ParsingBackSlash):
			return scan<State::ParsingBackSlash>();
		case (State::ParsingId):
			return scan<State::ParsingId>();
		case (State::ParsingQId):
			return scan<State::ParsingQId>();
		case (State::ParsingIdBackSlash):
			return scan<State::ParsingIdBackSlash>();
		case (State::End):
			return Token::End;
		case (State::IgnoreNextChar):
			state = State::Normal;
			return Token::None;
	}

	error = "Unandled Lexer State";
	return Token::Error;
}
