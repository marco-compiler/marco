//
// Created by Ale on 21/06/2021.
//

#ifndef PARSER_LEXER_M_VF_VARIABLETRACKER_H
#define PARSER_LEXER_M_VF_VARIABLETRACKER_H

#include <string>
#include <list>
#include <iostream>

using namespace std;

/**
 * represents an array range, $ special character is '-1'
 */
class Range {

public:
    Range(int leftValue, int rightValue);

    int leftValue, rightValue;
};

/**
 * Keeps tracks of a single variable, array or derivative that has been specified by command line arguments.
 */
class VariableTracker {
public:

    VariableTracker(const string &name, const bool isArray, const bool isDerivative, const uint16_t dim);

    void setRanges(const list <Range> &ranges);

    const string &getName() const;

    const bool getIsArray() const;

    const bool getIsDerivative() const;

    const uint16_t getDim() const;

    const list <Range> &getRanges() const;

    void dump(void) const;

private:

    const string _name;
    const bool _isArray;
    const bool _isDerivative;
    const uint16_t _dim;

    //-1 means "all"
    list <Range> _ranges;


};


#endif //PARSER_LEXER_M_VF_VARIABLETRACKER_H
