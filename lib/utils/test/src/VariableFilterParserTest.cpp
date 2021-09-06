//
// Created by Ale on 06/09/2021.
//
#include <iostream>
#include "../../utils/include/modelica/utils/VariableFilter.h"
#include "../../utils/include/modelica/utils/VariableFilterParser.h"
#include "../../utils/include/modelica/utils/VariableTracker.h"

using namespace std;
using namespace modelica;

void testNormal() {

    cout << "** SIMPLE VARIABLES **" << endl;

    string commandLineInput = "x;x1;_x22;x_var;";
    modelica::VariableFilter vf = VariableFilter();
    modelica::VariableFilterParser parser = VariableFilterParser();
    parser.parseCommandLine(commandLineInput, vf);
    assert(vf.isBypass() == false); //variable filtering is been used
    assert(vf.checkTrackedIdentifier("x") &&
           vf.checkTrackedIdentifier("x1") &&
           vf.checkTrackedIdentifier("_x22") &&
           vf.checkTrackedIdentifier("x_var"));
    //Check if parsing of all informations is correct

    VariableTracker x = vf.lookupByIdentifier("x");
    assert(!(x.getIsArray() || x.getIsDerivative()) && x.getName() == "x");
    std::cout << "test: x is ok" << std::endl;

    VariableTracker x1 = vf.lookupByIdentifier("x1");
    assert(!(x1.getIsArray() || x1.getIsDerivative()) && x1.getName() == "x1");
    std::cout << "test: x1 is ok" << std::endl;

    VariableTracker _x22 = vf.lookupByIdentifier("_x22");
    assert(!(_x22.getIsArray() || _x22.getIsDerivative()) && _x22.getName() == "_x22");
    std::cout << "test: _x22 is ok" << std::endl;

    VariableTracker x_var = vf.lookupByIdentifier("x_var");
    assert(!(x_var.getIsArray() || x_var.getIsDerivative()) && x_var.getName() == "x_var");
    std::cout << "test: x_var is ok" << std::endl;

    vf.dump();


}

void testArray() {
    cout << "\n\n** ARRAYS VARIABLES **" << endl;

    string commandLineInput = "x[$:$];xArray[0:10];y[$:199];ann[3:$,2:10,$:55];const;";
    modelica::VariableFilter vf = VariableFilter();
    modelica::VariableFilterParser parser = VariableFilterParser();
    parser.parseCommandLine(commandLineInput, vf);
    assert(vf.isBypass() == false); //variable filtering is been used
    assert(vf.checkTrackedIdentifier("x") &&
           vf.checkTrackedIdentifier("xArray") &&
           vf.checkTrackedIdentifier("y") &&
           vf.checkTrackedIdentifier("ann") && vf.checkTrackedIdentifier("const"));

    assert(vf.lookupByIdentifier("x").getIsArray() && !vf.lookupByIdentifier("x").getIsDerivative());
    assert(vf.lookupByIdentifier("xArray").getIsArray() && !vf.lookupByIdentifier("xArray").getIsDerivative());
    assert(vf.lookupByIdentifier("y").getIsArray() && !vf.lookupByIdentifier("y").getIsDerivative());
    assert(vf.lookupByIdentifier("ann").getIsArray() && !vf.lookupByIdentifier("ann").getIsDerivative());
    assert(!vf.lookupByIdentifier("const").getIsArray() && !vf.lookupByIdentifier("const").getIsDerivative());

    VariableTracker x = vf.lookupByIdentifier("x");
    assert(x.getDim() == 1 && x.getRangeOfDimensionN(0).noLowerBound() && x.getRangeOfDimensionN(0).noUpperBound());

    VariableTracker xArray = vf.lookupByIdentifier("xArray");
    assert(xArray.getDim() == 1 && xArray.getRangeOfDimensionN(0).leftValue == 0 &&
           xArray.getRangeOfDimensionN(0).rightValue == 10);

    VariableTracker y = vf.lookupByIdentifier("y");
    assert(y.getDim() == 1 && y.getRangeOfDimensionN(0).noLowerBound() && y.getRangeOfDimensionN(0).rightValue == 199);

    //ann[3:$,2:10,$:55]
    VariableTracker ann = vf.lookupByIdentifier("ann");
    assert(ann.getDim() == 3 && ann.getRangeOfDimensionN(0).leftValue == 3 && ann.getRangeOfDimensionN(0).noUpperBound()
           && ann.getRangeOfDimensionN(1).leftValue == 2 && ann.getRangeOfDimensionN(1).rightValue == 10
           && ann.getRangeOfDimensionN(2).noLowerBound() && ann.getRangeOfDimensionN(2).rightValue == 55
    );

    vf.dump();
    //Check if parsing of all informations is correct


}

void testRegex() {
    string commandLineInput = "/[a-z]+/;";
    modelica::VariableFilter vf = VariableFilter();
    modelica::VariableFilterParser parser = VariableFilterParser();
    parser.parseCommandLine(commandLineInput, vf);
    assert(vf.matchesRegex("a"));
    assert(vf.matchesRegex("ab"));
    assert(vf.matchesRegex("aaazzzbchdjfndhdbcys"));
    assert(!vf.matchesRegex("!abCCdEEciaOOO"));
    assert(!vf.matchesRegex("abCCdEEc000iaOOO"));
    assert(!vf.matchesRegex(""));

    modelica::VariableFilter vf2 = VariableFilter();
    modelica::VariableFilterParser parser2 = VariableFilterParser();

    auto testRegex = regex("([a-z]+)([_.a-z0-9]*)([a-z0-9]+)(@)([a-z]+)([.a-z]+)([a-z]+)");
    assert(regex_match("alelisi@polimi.it", testRegex));
    parser2.parseCommandLine("/([a-z]+)([_.a-z0-9]*)([a-z0-9]+)(@)([a-z]+)([.a-z]+)([a-z]+)/;", vf2);
    assert(vf2.matchesRegex("ciro@postino.it"));
}
int main() {

    testNormal();
    testArray();
    testRegex();

}

