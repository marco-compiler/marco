#pragma once

#define EXPECT(Token)                                                          \
	if (auto e = expect(Token); !e)                                              \
	return e.takeError()

#define TRY(outVar, expression)                                                \
	auto outVar = expression;                                                    \
	if (!outVar)                                                                 \
	return outVar.takeError()
