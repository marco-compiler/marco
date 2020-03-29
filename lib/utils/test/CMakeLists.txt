include(GoogleTest)

ADD_EXECUTABLE(utilsTest src/DefaultTest.cpp src/LexerTest.cpp src/NumbersLexerTest.cpp src/IntervalTest.cpp src/IRangeTest.cpp src/ScopeGuard.cpp src/ThreadPoolTest.cpp)
ADD_EXECUTABLE(modelica::utilsTest ALIAS utilsTest)

TARGET_LINK_LIBRARIES(utilsTest PRIVATE gtest gtest_main modelica::utils)
TARGET_INCLUDE_DIRECTORIES(utilsTest PUBLIC include PRIVATE src)
TARGET_COMPILE_FEATURES(utilsTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      utilsTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)


