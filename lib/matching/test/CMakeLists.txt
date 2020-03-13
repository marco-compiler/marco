include(GoogleTest)

add_executable(matchingTest src/MatchingTest.cpp src/VVarDependencyGraphTest.cpp)
add_executable(modelica::matchingTest ALIAS matchingTest) 
target_link_libraries(matchingTest PRIVATE gtest gtest_main modelica::matching)
target_include_directories(matchingTest PUBLIC include PRIVATE src)
target_compile_features(matchingTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      matchingTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

