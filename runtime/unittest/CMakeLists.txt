add_custom_target(MARCORuntimeUnitTests)

set_target_properties(MARCORuntimeUnitTests
    PROPERTIES FOLDER "MARCO runtime unit tests")

include(GoogleTest)

marco_add_unittest(MARCORuntimeTest
    BuiltInFunctionsTest.cpp
    MathTest.cpp
    UtilityFunctionsTest.cpp)

target_link_libraries(MARCORuntimeTest
    PUBLIC
    Support)
