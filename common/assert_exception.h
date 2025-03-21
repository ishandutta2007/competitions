#pragma once

#include <format>
#include <iostream>
#include <string>

/**
 * @brief Exception class for assertion failures.
 */
class AssertException {};

/**
 * @brief Asserts that a condition is true.
 *
 * @param condition The condition to check.
 * @throws AssertException if the condition is false.
 */
inline void Assert(bool condition) {
  if (!condition) {
    throw AssertException();
  }
}

/**
 * @brief Asserts that a condition is true, with a custom message.
 *
 * @param condition The condition to check.
 * @param message The message to display if the assertion fails.
 * @throws AssertException if the condition is false.
 */
inline void Assert(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << std::format("Assertion failed: {}\n", message);
    throw AssertException();
  }
}

/**
 * @brief Asserts that a condition is true, with a formatted message.
 *
 * @param condition The condition to check.
 * @param format The format string.
 * @param args The format arguments.
 * @throws AssertException if the condition is false.
 */
template <typename... Args>
inline void Assert(bool condition, std::format_string<Args...> format,
                   Args&&... args) {
  if (!condition) {
    std::string message = std::format(format, std::forward<Args>(args)...);
    std::cerr << std::format("Assertion failed: {}\n", message);
    throw AssertException();
  }
}
