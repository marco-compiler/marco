#pragma once

#define ASSERT_ERROR(expression, errorType)                                    \
	{                                                                            \
		Error error = expression;                                                  \
		bool handled = false;                                                      \
		handleAllErrors(                                                           \
				move(error), [&](const errorType& err) { handled = true; });           \
		ASSERT_TRUE(handled);                                                      \
	}

#define EXPECT_ERROR(expression, errorType)                                    \
	{                                                                            \
		Error error = expression;                                                  \
		bool handled = false;                                                      \
		handleAllErrors(                                                           \
				move(error), [&](const errorType& err) { handled = true; });           \
		EXPECT_TRUE(handled);                                                      \
	}
