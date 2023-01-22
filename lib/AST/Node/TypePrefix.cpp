#include "marco/AST/Node/TypePrefix.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const VariabilityQualifier& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(VariabilityQualifier qualifier)
	{
		switch (qualifier)
		{
			case VariabilityQualifier::discrete:
				return "discrete";
			case VariabilityQualifier::parameter:
				return "parameter";
			case VariabilityQualifier::constant:
				return "constant";
			case VariabilityQualifier::none:
				return "none";
		}

		return "unexpected";
	}

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const IOQualifier& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(IOQualifier qualifier)
	{
		switch (qualifier)
		{
			case IOQualifier::input:
				return "input";
			case IOQualifier::output:
				return "output";
			case IOQualifier::none:
				return "none";
		}

		return "unexpected";
	}

  TypePrefix::TypePrefix(
      VariabilityQualifier variabilityQualifier, IOQualifier ioQualifier)
      : variabilityQualifier(variabilityQualifier), ioQualifier(ioQualifier)
  {
  }

  void TypePrefix::print(llvm::raw_ostream& os, size_t indents) const
  {
    os << "Prefix: ";

    if (variabilityQualifier == VariabilityQualifier::none &&
        ioQualifier == IOQualifier::none) {
      os << "none";
    } else {
      bool space = false;

      if (variabilityQualifier != VariabilityQualifier::none) {
        os << variabilityQualifier;
        space = true;
      }

      if (ioQualifier != IOQualifier::none) {
        if (space) {
          os << " ";
        }

        os << ioQualifier;
      }
    }

    os.indent(indents);
  }

  bool TypePrefix::isDiscrete() const
  {
    return variabilityQualifier == VariabilityQualifier::discrete;
  }

  bool TypePrefix::isParameter() const
  {
    return variabilityQualifier == VariabilityQualifier::parameter;
  }

  bool TypePrefix::isConstant() const
  {
    return variabilityQualifier == VariabilityQualifier::constant;
  }

  bool TypePrefix::isInput() const
  {
    return ioQualifier == IOQualifier::input;
  }

  bool TypePrefix::isOutput() const
  {
    return ioQualifier == IOQualifier::output;
  }

  TypePrefix TypePrefix::none()
  {
    return TypePrefix(VariabilityQualifier::none, IOQualifier::none);
  }
}
