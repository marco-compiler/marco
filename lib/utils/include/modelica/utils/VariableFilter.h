//
// Created by Ale on 21/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLEFILTER_H
#define PARSER_LEXER_M_VF_VARIABLEFILTER_H

#include <iostream>
#include <list>
#include <regex>
#include <string>
#include <unordered_map>

#include "VariableTracker.h"

using namespace std;

/**
 *  Keeps tracks of a variables, arrays, derivatives (and regex for matching) we
 * want to print.
 */
namespace modelica {
    class VariableFilter {
    public:
        [[nodiscard]] bool isBypass() const;

        void setBypass(bool bypass);

        void addVariable(VariableTracker var);
        void addDerivative(VariableTracker var);

        void addRegexString(string regex);

        void dump();

        /**
         *
         * @param identifier the string that will be matched with all the regexes
         * @return true if there is a stored regular expression that matches the
         * received identifier
         */
        bool matchesRegex(const string &identifier);

        /**
         *
         * @param identifier the variable identifier we want to query
         * @return the variable tracker associated with that variable
         */
        VariableTracker lookupByIdentifier(const string &identifier);

        bool checkTrackedIdentifier(const string &identifier);

        /**
         * Check if derivative of var must be printed
         * @param derivedVariableIdentifier a model variable identifier
         * @return true if by command line arguments is specified that the derivative of provided variable identifier must be printed
         */
        bool printDerivative(const string &derivedVariableIdentifier);

    private:
        std::unordered_map<string, VariableTracker> _variables;
        std::unordered_map<string, VariableTracker> _derivatives;

        list<string> _regex;
        bool _bypass = true;
    };
}     // namespace modelica

#endif    // PARSER_LEXER_M_VF_VARIABLEFILTER_H
